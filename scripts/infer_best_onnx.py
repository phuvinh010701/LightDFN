#!/usr/bin/env python3
"""
Run LightDeepFilterNet inference using an exported ONNX model.

The ONNX model has a fixed T (set at export time via --chunk-frames, default 512).
This script pads short audio and processes long audio in non-overlapping chunks.

Example:
  python3 scripts/infer_best_onnx.py \
    --onnx checkpoints/lightdfn.onnx \
    --audio datasets/tests/audio-file.mp3 \
    --out datasets/tests/audio-file_enhanced.wav

  # With CUDA execution provider:
  python3 scripts/infer_best_onnx.py \
    --onnx checkpoints/lightdfn.onnx \
    --audio datasets/tests/audio-file.mp3 \
    --provider CUDAExecutionProvider
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.configs.config import load_config
from src.dataloader.fft import _norm_alpha, _running_mean_norm_erb, _running_unit_norm
from src.utils.audio import load_audio_mono_resampled, save_audio_peak_normalized
from src.utils.erb import get_erb_filterbanks


def _stft_features(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    window: torch.Tensor,
    erb_fb: torch.Tensor,
    nb_spec: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (spec_noisy, feat_erb_norm, feat_spec) for the full waveform.

    Args:
        x: [1, 1, T_samples]
    Returns:
        spec_noisy:    [1, 1, T_frames, F, 2]
        feat_erb_norm: [1, 1, T_frames, E]
        feat_spec:     [1, 1, T_frames, F', 2]
    """
    stft = torch.stft(
        x.reshape(-1, x.shape[-1]),
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window=window,
        return_complex=True,
        center=False,
    )
    B = x.shape[0]
    F_bins, T_frames = stft.shape[-2], stft.shape[-1]
    stft = stft.view(B, 1, F_bins, T_frames)  # [B, 1, F, T]

    spec_noisy = torch.view_as_real(
        stft.permute(0, 1, 3, 2).contiguous()
    )  # [B, 1, T, F, 2]

    power = stft.abs().pow(2)
    feat_erb = power.permute(0, 1, 3, 2) @ erb_fb.to(dtype=power.dtype)  # [B, 1, T, E]
    feat_erb_db = (feat_erb + 1e-10).log10() * 10.0
    feat_erb_norm = _running_mean_norm_erb(feat_erb_db.squeeze(1), alpha).unsqueeze(
        1
    )  # [B, 1, T, E]

    spec_slice = stft[:, :, :nb_spec, :].squeeze(1)  # [B, F', T]
    feat_spec = torch.view_as_real(
        _running_unit_norm(spec_slice, alpha).permute(0, 2, 1).contiguous()
    ).unsqueeze(1)  # [B, 1, T, F', 2]

    return spec_noisy, feat_erb_norm, feat_spec


def _spec_to_audio(
    spec: torch.Tensor, fft_size: int, hop_size: int, window: torch.Tensor
) -> torch.Tensor:
    """Reconstruct waveform from complex spectrogram [B, 1, T, F, 2]."""
    spec_c = torch.view_as_complex(spec[:, 0].contiguous()).permute(
        0, 2, 1
    )  # [B, F, T]
    return torch.istft(
        spec_c,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window=window,
        center=True,
    )


def enhance_file(
    onnx_path: Path,
    audio_path: Path,
    out_path: Path,
    config_path: Path,
    provider: str = "CPUExecutionProvider",
) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required: uv pip install onnxruntime  (or onnxruntime-gpu)"
        )

    model_cfg, _aug_cfg, loader_cfg, _train_cfg, _loss_cfg = load_config(
        str(config_path)
    )
    sess = ort.InferenceSession(str(onnx_path), providers=[provider])

    # Read fixed chunk size from the ONNX model input shape
    chunk_frames: int = sess.get_inputs()[0].shape[2]
    print(
        f"ONNX chunk size: {chunk_frames} frames ({chunk_frames * model_cfg.hop_size / model_cfg.sr:.2f} s)"
    )

    erb_fb, _ = get_erb_filterbanks(
        model_cfg.sr, model_cfg.fft_size, model_cfg.nb_erb, model_cfg.min_nb_freqs
    )

    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    audio, sr = load_audio_mono_resampled(str(audio_path), model_cfg.sr)
    x = audio.unsqueeze(0)  # [1, 1, T_samples]
    total_samples = x.shape[-1]

    window = torch.hann_window(model_cfg.fft_size)
    alpha = _norm_alpha(model_cfg.sr, model_cfg.hop_size, model_cfg.norm_tau)
    fft, hop = model_cfg.fft_size, model_cfg.hop_size

    # Pad tail so (total - fft) is a multiple of hop (center=False STFT)
    remainder = (total_samples - fft) % hop
    x_padded = torch.nn.functional.pad(x, (0, (hop - remainder) % hop))

    # Compute all features at once so running normalisation is consistent
    spec_noisy, feat_erb_norm, feat_spec = _stft_features(
        x_padded, fft, hop, window, erb_fb, loader_cfg.nb_spec, alpha
    )
    T_frames = spec_noisy.shape[2]

    enhanced_specs = []
    t0 = time.time()

    for chunk_start in range(0, T_frames, chunk_frames):
        chunk_end = min(chunk_start + chunk_frames, T_frames)
        n = chunk_end - chunk_start

        spec_c = spec_noisy[:, :, chunk_start:chunk_end]
        erb_c = feat_erb_norm[:, :, chunk_start:chunk_end]
        fspec_c = feat_spec[:, :, chunk_start:chunk_end]

        if n < chunk_frames:
            pad = chunk_frames - n
            spec_c = torch.nn.functional.pad(spec_c, (0, 0, 0, 0, 0, pad))
            erb_c = torch.nn.functional.pad(erb_c, (0, 0, 0, pad))
            fspec_c = torch.nn.functional.pad(fspec_c, (0, 0, 0, 0, 0, pad))

        out = sess.run(
            ["spec_e"],
            {
                "spec": spec_c.numpy(),
                "feat_erb": erb_c.numpy(),
                "feat_spec": fspec_c.numpy(),
            },
        )
        enhanced_specs.append(out[0][:, :, :n])

    elapsed = time.time() - t0
    print(
        f"ONNX inference: {elapsed:.2f} s  (RTF={elapsed / (total_samples / model_cfg.sr):.3f})"
    )

    import numpy as np

    enhanced = _spec_to_audio(
        torch.from_numpy(np.concatenate(enhanced_specs, axis=2)).nan_to_num(
            nan=0.0, posinf=0.0, neginf=0.0
        ),
        fft,
        hop,
        window,
    ).squeeze(0)

    if enhanced.numel() > total_samples:
        enhanced = enhanced[:total_samples]
    elif enhanced.numel() < total_samples:
        enhanced = torch.nn.functional.pad(
            enhanced, (0, total_samples - enhanced.numel())
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio_peak_normalized(enhanced, str(out_path), sr)
    print(f"Saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LightDFN ONNX inference")
    parser.add_argument("--onnx", type=str, default="checkpoints/lightdfn.onnx")
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--config", type=str, default="src/configs/default.yaml")
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider (CPUExecutionProvider / CUDAExecutionProvider)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    out_path = (
        Path(args.out)
        if args.out
        else audio_path.with_name(audio_path.stem + "_enhanced.wav")
    )

    enhance_file(
        onnx_path=Path(args.onnx),
        audio_path=audio_path,
        out_path=out_path,
        config_path=Path(args.config),
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
