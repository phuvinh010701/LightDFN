#!/usr/bin/env python3
"""
Run LightDeepFilterNet inference using an exported ONNX model.

The ONNX model accepts a fixed number of time frames (set at export time,
default 512 ≈ 5 s at 48 kHz / hop 480).  This script automatically pads
short audio and processes long audio in non-overlapping chunks.

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
from pathlib import Path
import time

import numpy as np
import torch
import torchaudio

from src.configs.config import load_config
from src.dataloader.fft import _norm_alpha, _running_mean_norm_erb, _running_unit_norm
from src.utils.erb import get_erb_filterbanks
from src.utils.io import resample


def _spec_to_audio(spec: torch.Tensor, fft_size: int, hop_size: int, window: torch.Tensor) -> torch.Tensor:
    """Reconstruct waveform from complex spectrogram tensor.

    Args:
        spec: Shape [B, 1, T_frames, F, 2] real/imag.
    Returns:
        Waveform: [B, T_samples]
    """
    spec_c = torch.view_as_complex(spec[:, 0].contiguous())  # [B, T, F]
    spec_c = spec_c.permute(0, 2, 1)  # [B, F, T]
    B = spec_c.shape[0]
    audio = torch.istft(
        spec_c.reshape(B, spec_c.shape[-2], spec_c.shape[-1]),
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window=window,
        center=True,
    )
    return audio


def _stft_features(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    window: torch.Tensor,
    erb_fb: torch.Tensor,
    nb_spec: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (spec_noisy, feat_erb_norm, feat_spec) from a waveform chunk.

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

    spec_noisy = torch.view_as_real(stft.permute(0, 1, 3, 2).contiguous())  # [B, 1, T, F, 2]

    power = stft.abs().pow(2)
    feat_erb = (power.permute(0, 1, 3, 2) @ erb_fb.to(dtype=power.dtype))  # [B, 1, T, E]
    feat_erb_db = (feat_erb + 1e-10).log10() * 10.0
    feat_erb_norm = _running_mean_norm_erb(feat_erb_db.squeeze(1), alpha).unsqueeze(1)

    spec_slice = stft[:, :, :nb_spec, :].squeeze(1)  # [B, F', T]
    spec_normed = _running_unit_norm(spec_slice, alpha)
    feat_spec = torch.view_as_real(spec_normed.permute(0, 2, 1).contiguous()).unsqueeze(1)  # [B, 1, T, F', 2]

    return spec_noisy, feat_erb_norm, feat_spec


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
        raise ImportError("onnxruntime is required: pip install onnxruntime  (or onnxruntime-gpu)")

    model_cfg, _aug_cfg, loader_cfg, _train_cfg, _loss_cfg = load_config(str(config_path))

    sess = ort.InferenceSession(str(onnx_path), providers=[provider])

    # Read the fixed chunk size from the ONNX model's input shape
    chunk_frames: int = sess.get_inputs()[0].shape[2]  # T dim of 'spec' input
    print(f"ONNX chunk size: {chunk_frames} frames  "
          f"({chunk_frames * model_cfg.hop_size / model_cfg.sr:.2f} s)")

    erb_fb, _ = get_erb_filterbanks(
        model_cfg.sr, model_cfg.fft_size, model_cfg.nb_erb, model_cfg.min_nb_freqs
    )

    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    audio, sr = torchaudio.load(str(audio_path))
    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(torch.float32)

    if sr != model_cfg.sr:
        audio = resample(audio, sr, model_cfg.sr, method="kaiser_best")
        sr = model_cfg.sr

    x = audio.unsqueeze(0)  # [1, 1, T_samples]
    window = torch.hann_window(model_cfg.fft_size)
    alpha = _norm_alpha(model_cfg.sr, model_cfg.hop_size, model_cfg.norm_tau)

    # With center=False: n_frames = (n_samples - fft_size) // hop_size + 1
    # => n_samples = (chunk_frames - 1) * hop_size + fft_size
    samples_per_chunk = (chunk_frames - 1) * model_cfg.hop_size + model_cfg.fft_size

    print(f"samples_per_chunk: {samples_per_chunk}")
    total_samples = x.shape[-1]

    # Pad audio so length is a multiple of samples_per_chunk
    remainder = total_samples % samples_per_chunk
    pad_samples = (samples_per_chunk - remainder) % samples_per_chunk
    if pad_samples > 0:
        x_padded = torch.nn.functional.pad(x, (0, pad_samples))
    else:
        x_padded = x

    n_chunks = x_padded.shape[-1] // samples_per_chunk

    enhanced_chunks: list[torch.Tensor] = []

    for i in range(n_chunks):
        start = time.time()
        chunk = x_padded[:, :, i * samples_per_chunk : (i + 1) * samples_per_chunk]

        spec_noisy, feat_erb_norm, feat_spec = _stft_features(
            chunk, model_cfg.fft_size, model_cfg.hop_size, window,
            erb_fb, loader_cfg.nb_spec, alpha,
        )

        feeds = {
            "spec":      spec_noisy.numpy(),
            "feat_erb":  feat_erb_norm.numpy(),
            "feat_spec": feat_spec.numpy(),
        }
        outputs = sess.run(["enhanced_spec"], feeds)
        enhanced_spec = torch.from_numpy(outputs[0])  # [1, 1, T_frames, F, 2]

        chunk_audio = _spec_to_audio(
            enhanced_spec, model_cfg.fft_size, model_cfg.hop_size, window
        )  # [1, T_samples_out]
        enhanced_chunks.append(chunk_audio)
        end = time.time()
        print(f"Chunk {i} time: {end - start} seconds")

    enhanced = torch.cat(enhanced_chunks, dim=-1).squeeze(0)  # [T_samples_total]

    # Trim to original length
    if enhanced.numel() > total_samples:
        enhanced = enhanced[:total_samples]
    elif enhanced.numel() < total_samples:
        enhanced = torch.nn.functional.pad(enhanced, (0, total_samples - enhanced.numel()))

    wav = enhanced.detach().numpy().astype(np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = wav / max(peak, 1e-8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), torch.from_numpy(wav).unsqueeze(0), sr)
    print(f"Saved enhanced audio → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LightDFN ONNX inference")
    parser.add_argument("--onnx", type=str, default="checkpoints/lightdfn.onnx", help="ONNX model path")
    parser.add_argument("--audio", type=str, required=True, help="Input audio file (.mp3/.wav/...)")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output wav path. Default: <input_stem>_enhanced.wav in same folder.",
    )
    parser.add_argument("--config", type=str, default="src/configs/default.yaml", help="Model config YAML")
    parser.add_argument(
        "--provider",
        type=str,
        default="CPUExecutionProvider",
        help="ONNX Runtime execution provider (default: CPUExecutionProvider; use CUDAExecutionProvider for GPU)",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    audio_path = Path(args.audio)
    if args.out is None:
        out_path = audio_path.with_name(audio_path.stem + "_enhanced.wav")
    else:
        out_path = Path(args.out)

    enhance_file(
        onnx_path=onnx_path,
        audio_path=audio_path,
        out_path=out_path,
        config_path=Path(args.config),
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
