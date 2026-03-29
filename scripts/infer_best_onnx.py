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
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

from src.configs.config import load_config
from src.dataloader.fft import _norm_alpha, _running_mean_norm_erb, _running_unit_norm
from src.utils.erb import get_erb_filterbanks
from src.utils.io import resample


def _spec_to_audio(
    spec: torch.Tensor, fft_size: int, hop_size: int, window: torch.Tensor
) -> torch.Tensor:
    """Reconstruct waveform from complex spectrogram tensor.

    Args:
        spec: Shape [B, 1, T_frames, F, 2] real/imag.
    Returns:
        Waveform: [B, T_samples]
    """
    spec_c = torch.view_as_complex(spec[:, 0].contiguous())  # [B, T, F]
    spec_c = spec_c.permute(0, 2, 1)  # [B, F, T]
    B = spec_c.shape[0]
    # center=True is required: with center=False the Hann window is 0 at position 0,
    # making the OLA envelope 0 at the first sample and failing PyTorch's COLA check.
    # center=True trims n_fft//2 samples from both ends of the OLA result.
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

    spec_noisy = torch.view_as_real(
        stft.permute(0, 1, 3, 2).contiguous()
    )  # [B, 1, T, F, 2]

    power = stft.abs().pow(2)
    feat_erb = power.permute(0, 1, 3, 2) @ erb_fb.to(dtype=power.dtype)  # [B, 1, T, E]
    feat_erb_db = (feat_erb + 1e-10).log10() * 10.0
    feat_erb_norm = _running_mean_norm_erb(feat_erb_db.squeeze(1), alpha).unsqueeze(1)

    spec_slice = stft[:, :, :nb_spec, :].squeeze(1)  # [B, F', T]
    spec_normed = _running_unit_norm(spec_slice, alpha)
    feat_spec = torch.view_as_real(spec_normed.permute(0, 2, 1).contiguous()).unsqueeze(
        1
    )  # [B, 1, T, F', 2]

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
        raise ImportError(
            "onnxruntime is required: pip install onnxruntime  (or onnxruntime-gpu)"
        )

    model_cfg, _aug_cfg, loader_cfg, _train_cfg, _loss_cfg = load_config(
        str(config_path)
    )

    sess = ort.InferenceSession(str(onnx_path), providers=[provider])

    # Read the fixed chunk size from the ONNX model's input shape
    chunk_frames: int = sess.get_inputs()[0].shape[2]  # T dim of 'spec' input
    print(
        f"ONNX chunk size: {chunk_frames} frames  "
        f"({chunk_frames * model_cfg.hop_size / model_cfg.sr:.2f} s)"
    )

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
    total_samples = x.shape[-1]

    # Pad tail so (total - fft) is a multiple of hop (center=False STFT requirement)
    fft = model_cfg.fft_size
    hop = model_cfg.hop_size
    remainder = (total_samples - fft) % hop
    pad_tail = (hop - remainder) % hop
    x_padded = torch.nn.functional.pad(x, (0, pad_tail))  # [1, 1, T_padded]

    # Compute ALL STFT features at once so running normalisation is correct over the full
    # signal.  Splitting into chunks and calling _stft_features per chunk would reset the
    # running state each time, causing incorrect feature statistics for every chunk after
    # the first.
    spec_noisy_all, feat_erb_norm_all, feat_spec_all = _stft_features(
        x_padded, fft, hop, window, erb_fb, loader_cfg.nb_spec, alpha
    )
    # spec_noisy_all:    [1, 1, T_frames, F, 2]
    # feat_erb_norm_all: [1, 1, T_frames, E]
    # feat_spec_all:     [1, 1, T_frames, F', 2]
    T_frames = spec_noisy_all.shape[2]

    # Feed to ONNX in chunk_frames-sized batches.  The chunked ONNX model has no
    # explicit state I/O — each chunk starts with zero-initialised GRU states,
    # which matches the training regime (short clips, zero-init state).
    # All enhanced frames are collected and reconstructed in one ISTFT call to avoid
    # edge artefacts from per-chunk ISTFT.
    enhanced_specs: list[np.ndarray] = []
    t0 = time.time()

    for chunk_start in range(0, T_frames, chunk_frames):
        chunk_end = min(chunk_start + chunk_frames, T_frames)
        n = chunk_end - chunk_start

        spec_chunk = spec_noisy_all[:, :, chunk_start:chunk_end, :, :]
        erb_chunk = feat_erb_norm_all[:, :, chunk_start:chunk_end, :]
        fspec_chunk = feat_spec_all[:, :, chunk_start:chunk_end, :, :]

        # Pad the last chunk to chunk_frames if it is shorter
        if n < chunk_frames:
            pad = chunk_frames - n
            spec_chunk = torch.nn.functional.pad(spec_chunk, (0, 0, 0, 0, 0, pad))
            erb_chunk = torch.nn.functional.pad(erb_chunk, (0, 0, 0, pad))
            fspec_chunk = torch.nn.functional.pad(fspec_chunk, (0, 0, 0, 0, 0, pad))

        feeds = {
            "spec": spec_chunk.numpy(),
            "feat_erb": erb_chunk.numpy(),
            "feat_spec": fspec_chunk.numpy(),
        }
        outputs = sess.run(["enhanced_spec"], feeds)
        out_spec = outputs[0]  # [1, 1, chunk_frames, F, 2]
        # Discard padding frames at the tail of the last chunk
        if n < chunk_frames:
            out_spec = out_spec[:, :, :n, :, :]
        enhanced_specs.append(out_spec)

    elapsed = time.time() - t0
    audio_dur = total_samples / model_cfg.sr
    print(
        f"Processed {T_frames} frames in {elapsed:.2f} s  "
        f"(audio={audio_dur:.2f} s, RTF={elapsed / audio_dur:.3f})"
    )

    # Reconstruct waveform from all enhanced frames in ONE ISTFT call — avoids
    # per-chunk edge artefacts and centre=True timing issues with very short chunks.
    all_specs = np.concatenate(enhanced_specs, axis=2)  # [1, 1, T_frames, F, 2]
    enhanced = _spec_to_audio(
        torch.from_numpy(all_specs).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0),
        fft,
        hop,
        window,
    ).squeeze(0)

    # Trim to original length
    if enhanced.numel() > total_samples:
        enhanced = enhanced[:total_samples]
    elif enhanced.numel() < total_samples:
        enhanced = torch.nn.functional.pad(
            enhanced, (0, total_samples - enhanced.numel())
        )

    wav = enhanced.detach().numpy().astype(np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = wav / max(peak, 1e-8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), torch.from_numpy(wav).unsqueeze(0), sr)
    print(f"Saved enhanced audio → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LightDFN ONNX inference")
    parser.add_argument(
        "--onnx", type=str, default="checkpoints/lightdfn.onnx", help="ONNX model path"
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Input audio file (.mp3/.wav/...)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output wav path. Default: <input_stem>_enhanced.wav in same folder.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/default.yaml",
        help="Model config YAML",
    )
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
