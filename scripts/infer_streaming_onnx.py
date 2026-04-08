#!/usr/bin/env python3
"""
Run LightDeepFilterNet streaming ONNX inference (one hop = 480 samples per call).

Uses the streaming model exported with --streaming flag, which exposes GRU
hidden states as explicit inputs/outputs and processes T=1 frame per call.

Features (spec, feat_erb, feat_spec) are computed *per-frame* with running
normalisation state maintained across calls — matching true real-time streaming
semantics where only the current audio chunk is available at each step.

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
from src.dataloader.fft import _norm_alpha
from src.utils.erb import get_erb_filterbanks
from src.utils.io import resample


def _compute_frame_features(
    frame_samples: torch.Tensor,
    fft_size: int,
    hop_size: int,
    window: torch.Tensor,
    erb_fb: torch.Tensor,
    nb_spec: int,
    erb_state: torch.Tensor,
    unit_state: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute STFT features for a single frame and update running normalisation state.

    This is the per-frame version of _stft_features used in true streaming mode:
    only the current audio chunk (``fft_size`` samples) is required, and running
    statistics are carried across calls via ``erb_state`` / ``unit_state``.

    Args:
        frame_samples: ``[1, fft_size]`` — overlap buffer + current hop samples.
        fft_size:      FFT size (= window length).
        hop_size:      Hop size (used only to set torch.stft hop_length).
        window:        Hann window tensor of length ``fft_size``.
        erb_fb:        ERB filterbank ``[F, nb_erb]``.
        nb_spec:       Number of complex spec bins for feat_spec.
        erb_state:     Running mean state ``[1, nb_erb]`` (updated in-place logically).
        unit_state:    Running unit-norm state ``[1, nb_spec]`` (updated in-place logically).
        alpha:         Exponential smoothing factor.

    Returns:
        spec_noisy:     ``[1, 1, 1, F, 2]``
        feat_erb:       ``[1, 1, 1, nb_erb]``
        feat_spec:      ``[1, 1, 1, nb_spec, 2]``
        new_erb_state:  ``[1, nb_erb]``  — updated running mean state
        new_unit_state: ``[1, nb_spec]`` — updated running unit-norm state
    """
    one_minus_alpha = 1.0 - alpha

    # STFT: [1, fft_size] → [1, F, 1] (single frame)
    stft = torch.stft(
        frame_samples,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window=window,
        return_complex=True,
        center=False,
    )  # [1, F, 1]

    # spec_noisy [1, 1, 1, F, 2]
    spec_c = stft[:, :, 0]  # [1, F]
    spec_noisy = torch.view_as_real(spec_c).unsqueeze(0).unsqueeze(2)  # [1, 1, 1, F, 2]

    # ERB power features
    power = stft.abs().pow(2)[:, :, 0]  # [1, F]
    erb_power = power @ erb_fb.to(dtype=power.dtype)  # [1, nb_erb]
    erb_db = (erb_power + 1e-10).log10() * 10.0  # [1, nb_erb]

    # Running mean normalisation (matches Rust band_mean_norm_erb)
    new_erb_state = erb_db * one_minus_alpha + erb_state * alpha
    feat_erb_norm = (erb_db - new_erb_state) / 40.0  # [1, nb_erb]
    feat_erb = feat_erb_norm.unsqueeze(0).unsqueeze(2)  # [1, 1, 1, nb_erb]

    # Complex spec features (first nb_spec bins)
    spec_slice = stft[:, :nb_spec, 0]  # [1, nb_spec] complex

    # Running unit normalisation (matches Rust band_unit_norm)
    new_unit_state = spec_slice.abs() * one_minus_alpha + unit_state * alpha
    spec_normed = spec_slice / new_unit_state.sqrt()  # [1, nb_spec] complex
    feat_spec = (
        torch.view_as_real(spec_normed).unsqueeze(0).unsqueeze(2)
    )  # [1, 1, 1, nb_spec, 2]

    return spec_noisy, feat_erb, feat_spec, new_erb_state, new_unit_state


def _spec_to_audio(
    spec: torch.Tensor, fft_size: int, hop_size: int, window: torch.Tensor
) -> torch.Tensor:
    """[B, 1, T_frames, F, 2] -> [B, T_samples]"""
    spec_c = torch.view_as_complex(spec[:, 0].contiguous())  # [B, T, F]
    spec_c = spec_c.permute(0, 2, 1)  # [B, F, T]
    # center=True is required: with center=False the Hann window is 0 at position 0,
    # making the OLA envelope 0 at the first sample and failing PyTorch's COLA check.
    # center=True trims n_fft//2 samples from both ends of the OLA result, which
    # introduces a ~10 ms (480-sample) start-of-file shift — acceptable for enhancement.
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
        raise ImportError(
            "onnxruntime is required: pip install onnxruntime  (or onnxruntime-gpu)"
        )

    model_cfg, _aug_cfg, loader_cfg, _train_cfg, _loss_cfg = load_config(
        str(config_path)
    )
    hop = model_cfg.hop_size
    fft = model_cfg.fft_size

    sess = ort.InferenceSession(str(onnx_path), providers=[provider])

    # Verify this is the streaming model (T=1 inputs, hidden state + conv buffer I/O)
    input_names = [i.name for i in sess.get_inputs()]
    if (
        "h_enc" not in input_names
        or "buf_erb0" not in input_names
        or "buf_spec" not in input_names
    ):
        raise ValueError(
            "This does not look like a streaming model (or it's an old export). "
            "Re-export with: python3 scripts/export_onnx.py --streaming"
        )

    erb_fb, _ = get_erb_filterbanks(
        model_cfg.sr, fft, model_cfg.nb_erb, model_cfg.min_nb_freqs
    )  # erb_fb: [F, nb_erb]

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
    x = torch.nn.functional.pad(audio, (0, pad_tail))  # [1, T_padded]

    T_frames = (x.shape[-1] - fft) // hop + 1

    # ---- Initialise GRU hidden states and conv buffers to zero ----
    def _zero_input(name: str) -> np.ndarray:
        meta = next(i for i in sess.get_inputs() if i.name == name)
        return np.zeros(meta.shape, dtype=np.float32)

    state_enc = _zero_input("h_enc")
    state_erb = _zero_input("h_erb")
    state_df = _zero_input("h_df")
    buf_erb0 = _zero_input("buf_erb0")
    buf_df0 = _zero_input("buf_df0")
    buf_dfp = _zero_input("buf_dfp")
    buf_spec = _zero_input("buf_spec")

    # ---- Initialise running normalisation states ----
    # These mirror the initial values in _running_mean_norm_erb and _running_unit_norm.
    nb_erb = erb_fb.shape[1]
    nb_spec = loader_cfg.nb_spec
    erb_state = torch.linspace(-60.0, -90.0, nb_erb).unsqueeze(0)  # [1, nb_erb]
    unit_state = torch.linspace(0.001, 0.0001, nb_spec).unsqueeze(0)  # [1, nb_spec]

    # ---- Audio overlap buffer for per-frame STFT (center=False) ----
    # Each STFT frame needs fft_size samples: (fft - hop) past samples + hop new samples.
    overlap = fft - hop
    audio_buf = torch.zeros(1, overlap)  # [1, overlap]

    # ---- Process one audio chunk (hop samples) at a time ----
    # GRU hidden states are carried across the entire file (true streaming semantics).
    # Conv buffers (buf_*) maintain causal context continuously.
    LOG_EVERY = 500

    enhanced_specs: list[np.ndarray] = []
    t0 = time.time()

    for i in range(T_frames):
        # Gather the next hop of new samples
        new_samples = x[:, i * hop : i * hop + hop]  # [1, hop]

        # Build the fft_size-sample window for this STFT frame
        frame_samples = torch.cat([audio_buf, new_samples], dim=-1)  # [1, fft]

        # Slide the audio overlap buffer forward
        audio_buf = frame_samples[:, -overlap:]

        # Compute spec / feat_erb / feat_spec for this single frame, updating running states
        spec_noisy_f, feat_erb_f, feat_spec_f, erb_state, unit_state = (
            _compute_frame_features(
                frame_samples,
                fft,
                hop,
                window,
                erb_fb,
                nb_spec,
                erb_state,
                unit_state,
                alpha,
            )
        )

        # Feed to ONNX model — no look-ahead offset: streaming model already drops pad_feat
        feeds = {
            "spec": spec_noisy_f.numpy(),
            "feat_erb": feat_erb_f.numpy(),
            "feat_spec": feat_spec_f.numpy(),
            "h_enc": state_enc,
            "h_erb": state_erb,
            "h_df": state_df,
            "buf_erb0": buf_erb0,
            "buf_df0": buf_df0,
            "buf_dfp": buf_dfp,
            "buf_spec": buf_spec,
        }
        outputs = sess.run(None, feeds)
        enhanced_specs.append(outputs[0])  # [1, 1, 1, F, 2]
        state_enc = outputs[1]
        state_erb = outputs[2]
        state_df = outputs[3]
        buf_erb0 = outputs[4]
        buf_df0 = outputs[5]
        buf_dfp = outputs[6]
        buf_spec = outputs[7]

        if i % LOG_EVERY == 0:
            print(f"Frame {i}/{T_frames}")

    elapsed = time.time() - t0
    audio_dur = total_samples / model_cfg.sr
    print(
        f"Processed {T_frames} frames in {elapsed:.2f} s  "
        f"(audio={audio_dur:.2f} s, RTF={elapsed / audio_dur:.3f})"
    )

    # Reconstruct waveform from all enhanced STFT frames in one ISTFT call
    all_specs = np.concatenate(enhanced_specs, axis=2)  # [1, 1, T, F, 2]
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
    parser = argparse.ArgumentParser(
        description="LightDFN streaming ONNX inference (1 frame/call)"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="checkpoints/lightdfn_streaming.onnx",
        help="Streaming ONNX model path",
    )
    parser.add_argument("--audio", type=str, required=True, help="Input audio file")
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
        help="ONNX Runtime execution provider (default: CPUExecutionProvider)",
    )
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    audio_path = Path(args.audio)
    out_path = (
        Path(args.out)
        if args.out
        else audio_path.with_name(audio_path.stem + "_enhanced.wav")
    )

    enhance_file(
        onnx_path=onnx_path,
        audio_path=audio_path,
        out_path=out_path,
        config_path=Path(args.config),
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
