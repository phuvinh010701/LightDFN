#!/usr/bin/env python3
"""
Run LightDeepFilterNet inference on a single audio file.

Example:
  python3 scripts/infer_best.py \
    --ckpt checkpoints/best.pt \
    --audio datasets/tests/noisy.mp3 \
    --out datasets/tests/noisy_enhanced.wav
"""

import argparse
from pathlib import Path

import torch

from src.configs.config import load_config
from src.dataloader.fft import _norm_alpha, _running_mean_norm_erb, _running_unit_norm
from src.model.lightdeepfilternet import init_model
from src.utils.audio import (
    compute_stft,
    load_audio_mono_resampled,
    save_audio_peak_normalized,
    spec_to_audio,
)
from src.utils.io import get_device


@torch.no_grad()
def enhance_file(
    ckpt_path: Path,
    audio_path: Path,
    out_path: Path,
    config_path: Path,
) -> None:
    model_cfg, _aug_cfg, loader_cfg, _train_cfg, _loss_cfg = load_config(
        str(config_path)
    )

    device = get_device()
    model_cfg.batch_size = 1
    model = init_model(model_cfg).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    audio, sr = load_audio_mono_resampled(str(audio_path), model_cfg.sr)

    x = audio.unsqueeze(0).to(device)  # [1, 1, T]
    window = torch.hann_window(model_cfg.fft_size, device=device)

    spec_noisy = compute_stft(
        x, model_cfg.fft_size, model_cfg.hop_size, window
    )  # [B, C, T, F, 2]
    stft = torch.view_as_complex(spec_noisy.contiguous()).permute(
        0, 1, 3, 2
    )  # [B, C, F, T]

    # ERB features
    erb_fb = model.erb_fb.to(device=device, dtype=stft.real.dtype)  # [F, E]
    power = stft.abs().pow(2)  # [B, C, F, T]
    feat_erb = power.permute(0, 1, 3, 2) @ erb_fb  # [B, C, T, E]
    feat_erb_db = (feat_erb + 1e-10).log10() * 10.0  # [B, C, T, E]

    alpha = _norm_alpha(model_cfg.sr, model_cfg.hop_size, model_cfg.norm_tau)
    feat_erb_norm = _running_mean_norm_erb(feat_erb_db.squeeze(1), alpha).unsqueeze(
        1
    )  # [B, 1, T, E]

    # Spec features
    spec_slice = stft[:, :, : loader_cfg.nb_spec, :].squeeze(1)  # [B, F', T]
    spec_normed = _running_unit_norm(spec_slice, alpha)  # [B, F', T] complex
    feat_spec = torch.view_as_real(spec_normed.permute(0, 2, 1).contiguous()).unsqueeze(
        1
    )  # [B, 1, T, F', 2]

    # Forward pass on the full sequence
    enhanced_spec, *_ = model(spec_noisy, feat_erb_norm, feat_spec)

    # Back to waveform
    enhanced = spec_to_audio(
        enhanced_spec.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0),
        model_cfg.fft_size,
        model_cfg.hop_size,
        window,
    )
    enhanced = enhanced.squeeze(0).detach().cpu()

    # Trim/pad to original length
    target_len = x.shape[-1]
    if enhanced.numel() > target_len:
        enhanced = enhanced[:target_len]
    elif enhanced.numel() < target_len:
        enhanced = torch.nn.functional.pad(enhanced, (0, target_len - enhanced.numel()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_audio_peak_normalized(enhanced, str(out_path), sr)


def main() -> None:
    parser = argparse.ArgumentParser(description="LightDFN inference from best.pt")
    parser.add_argument(
        "--ckpt", type=str, default="checkpoints/best.pt", help="Checkpoint path"
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
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    audio_path = Path(args.audio)
    if args.out is None:
        out_path = audio_path.with_name(audio_path.stem + "_enhanced.wav")
    else:
        out_path = Path(args.out)

    enhance_file(
        ckpt_path=ckpt_path,
        audio_path=audio_path,
        out_path=out_path,
        config_path=Path(args.config),
    )


if __name__ == "__main__":
    main()
