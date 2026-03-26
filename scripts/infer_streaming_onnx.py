#!/usr/bin/env python3
"""
Run LightDeepFilterNet streaming ONNX inference (one hop = 480 samples per call).

Uses the streaming model exported with --streaming flag, which exposes LiGRU
hidden states as explicit inputs/outputs and processes T=1 frame per call.

STFT features are computed for the whole file at once so that the running
normalisation (ERB mean, unit-norm) is identical to the offline approach.
The model then processes one frame at a time with GRU hidden-state threading —
matching real-time streaming semantics while preserving audio quality.

Example:
  # Export first:
  python3 scripts/export_onnx.py --ckpt checkpoints/best.pt \
    --out checkpoints/lightdfn_streaming.onnx --streaming

  # Then infer:
  python3 scripts/infer_streaming_onnx.py \
    --onnx checkpoints/lightdfn_streaming.onnx \
    --audio datasets/tests/audio-file.mp3

  # With GPU:
  python3 scripts/infer_streaming_onnx.py \
    --onnx checkpoints/lightdfn_streaming.onnx \
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


def _stft_features(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    window: torch.Tensor,
    erb_fb: torch.Tensor,
    nb_spec: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (spec_noisy, feat_erb_norm, feat_spec) from a waveform.

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


def _spec_to_audio(spec: torch.Tensor, fft_size: int, hop_size: int, window: torch.Tensor) -> torch.Tensor:
    """[B, 1, T_frames, F, 2] -> [B, T_samples]"""
    spec_c = torch.view_as_complex(spec[:, 0].contiguous())  # [B, T, F]
    spec_c = spec_c.permute(0, 2, 1)                         # [B, F, T]
    audio = torch.istft(
        spec_c,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window=window,
        center=True,
    )
    return audio


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
    hop = model_cfg.hop_size
    fft = model_cfg.fft_size

    sess = ort.InferenceSession(str(onnx_path), providers=[provider])

    # Verify this is the streaming model (T=1 inputs, hidden state + conv buffer I/O)
    input_names = [i.name for i in sess.get_inputs()]
    if "h_enc" not in input_names or "buf_erb0" not in input_names or "buf_spec" not in input_names:
        raise ValueError(
            "This does not look like a streaming model (or it's an old export). "
            "Re-export with: python3 scripts/export_onnx.py --streaming"
        )

    erb_fb, _ = get_erb_filterbanks(
        model_cfg.sr, fft, model_cfg.nb_erb, model_cfg.min_nb_freqs
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

    window = torch.hann_window(fft)
    alpha = _norm_alpha(model_cfg.sr, hop, model_cfg.norm_tau)
    total_samples = audio.shape[-1]

    # Pad tail so (total - fft) is a multiple of hop (center=False STFT requirement)
    remainder = (total_samples - fft) % hop
    pad_tail = (hop - remainder) % hop
    x = torch.nn.functional.pad(audio.unsqueeze(0), (0, pad_tail))  # [1, 1, T_padded]

    # Compute all STFT features at once — running normalisation is correct over full signal
    spec_noisy, feat_erb_norm, feat_spec = _stft_features(
        x, fft, hop, window, erb_fb, loader_cfg.nb_spec, alpha
    )
    # spec_noisy:    [1, 1, T_frames, F, 2]
    # feat_erb_norm: [1, 1, T_frames, E]
    # feat_spec:     [1, 1, T_frames, F', 2]
    T_frames = spec_noisy.shape[2]

    # Initialise GRU hidden states and conv buffers to zero
    def _zero_input(name: str) -> np.ndarray:
        meta = next(i for i in sess.get_inputs() if i.name == name)
        return np.zeros(meta.shape, dtype=np.float32)

    state_enc  = _zero_input("h_enc")
    state_erb  = _zero_input("h_erb")
    state_df   = _zero_input("h_df")
    buf_erb0   = _zero_input("buf_erb0")
    buf_df0    = _zero_input("buf_df0")
    buf_dfp    = _zero_input("buf_dfp")
    buf_spec   = _zero_input("buf_spec")

    # The offline model applies pad_feat (ConstantPad2d with -conv_lookahead/+conv_lookahead)
    # to shift features 2 frames ahead before the encoder. Replicate this in streaming by
    # offsetting the feature index: use features at frame i+conv_lookahead for spec at frame i.
    conv_la = model_cfg.conv_lookahead  # typically 2

    # Process one frame at a time, threading the GRU hidden states
    enhanced_specs: list[np.ndarray] = []
    t0 = time.time()

    for i in range(T_frames):
        i_feat = min(i + conv_la, T_frames - 1)  # look-ahead feature index (clamped at end)
        feeds = {
            "spec":      spec_noisy[:, :, i : i + 1, :, :].numpy(),
            "feat_erb":  feat_erb_norm[:, :, i_feat : i_feat + 1, :].numpy(),
            "feat_spec": feat_spec[:, :, i_feat : i_feat + 1, :, :].numpy(),
            "h_enc":     state_enc,
            "h_erb":     state_erb,
            "h_df":      state_df,
            "buf_erb0":  buf_erb0,
            "buf_df0":   buf_df0,
            "buf_dfp":   buf_dfp,
            "buf_spec":  buf_spec,
        }
        outputs = sess.run(None, feeds)
        enhanced_specs.append(outputs[0])  # [1, 1, 1, F, 2]
        state_enc = outputs[1]
        state_erb = outputs[2]
        state_df  = outputs[3]
        buf_erb0  = outputs[4]
        buf_df0   = outputs[5]
        buf_dfp   = outputs[6]
        buf_spec  = outputs[7]

    elapsed = time.time() - t0
    audio_dur = total_samples / model_cfg.sr
    print(f"Processed {T_frames} frames in {elapsed:.2f} s  "
          f"(audio={audio_dur:.2f} s, RTF={elapsed/audio_dur:.3f})")

    # Reconstruct waveform from all enhanced STFT frames in one ISTFT call
    all_specs = np.concatenate(enhanced_specs, axis=2)  # [1, 1, T, F, 2]
    enhanced = _spec_to_audio(torch.from_numpy(all_specs), fft, hop, window).squeeze(0)

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
    parser = argparse.ArgumentParser(description="LightDFN streaming ONNX inference (1 frame/call)")
    parser.add_argument("--onnx", type=str, default="checkpoints/lightdfn_streaming.onnx", help="Streaming ONNX model path")
    parser.add_argument("--audio", type=str, required=True, help="Input audio file")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output wav path. Default: <input_stem>_enhanced.wav in same folder.",
    )
    parser.add_argument("--config", type=str, default="src/configs/default.yaml", help="Model config YAML")
    parser.add_argument(
        "--provider", type=str, default="CPUExecutionProvider",
        help="ONNX Runtime execution provider (default: CPUExecutionProvider)",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    audio_path = Path(args.audio)
    out_path = Path(args.out) if args.out else audio_path.with_name(audio_path.stem + "_enhanced.wav")

    enhance_file(
        onnx_path=onnx_path,
        audio_path=audio_path,
        out_path=out_path,
        config_path=Path(args.config),
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
