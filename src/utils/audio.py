import numpy as np
import torch
import torchaudio
from torch import Tensor

from src.utils.io import resample

_EPS = 1e-10


def mix_audio_signal(
    clean: Tensor,
    noise: Tensor,
    snr_db: float,
    gain_db: float,
    clean_distorted: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Mix clean speech and noise at a target SNR with anti-clipping protection.

    Returns ``(clean_out, noise_out, mixture)``.
    """
    gain_linear = 10.0 ** (gain_db / 20.0)
    clean_out: Tensor = clean * gain_linear
    clean_mix = (
        (clean_distorted * gain_linear)
        if clean_distorted is not None
        else clean_out.clone()
    )

    e_clean = clean_out.pow(2).sum().clamp(min=_EPS)
    e_noise = noise.pow(2).sum().clamp(min=_EPS)
    snr_linear = 10.0 ** (snr_db / 10.0)
    scale: Tensor = (e_clean / (e_noise * snr_linear)).sqrt()
    noise_out = noise * scale

    mixture = clean_mix + noise_out

    max_all = max(
        clean_out.abs().max().item(),
        noise_out.abs().max().item(),
        mixture.abs().max().item(),
    )
    if max_all - 1.0 > _EPS:
        sf = 1.0 / (max_all + _EPS)
        clean_out = clean_out * sf
        noise_out = noise_out * sf
        mixture = mixture * sf

    return clean_out, noise_out, mixture


def combine_noises(
    noises: list[Tensor],
    num_channels: int,
    target_length: int,
    rng: np.random.Generator,
) -> Tensor:
    """Combine a list of noise tensors into one (num_channels, target_length) tensor."""
    processed: list[Tensor] = []

    for noise in noises:
        c, t = noise.shape

        if t == 0:
            noise = torch.zeros(
                (c, target_length),
                dtype=noise.dtype,
                device=noise.device,
            )
        elif t < target_length:
            idx = torch.arange(target_length, device=noise.device) % t
            noise = noise[:, idx]
        elif t > target_length:
            start = int(rng.integers(0, t - target_length + 1))
            noise = noise[:, start : start + target_length]

        c = noise.shape[0]

        if c < num_channels:
            extra = num_channels - c
            extra_idx = rng.integers(0, c, size=extra)
            idx = np.concatenate([np.arange(c), extra_idx])
            noise = noise[torch.as_tensor(idx, device=noise.device)]
        elif c > num_channels:
            idx = np.sort(rng.choice(c, num_channels, replace=False))
            noise = noise[torch.as_tensor(idx, device=noise.device)]

        processed.append(noise)

    return torch.stack(processed, dim=0).mean(dim=0)


def adjust_channels_jointly(
    num_channels: int,
    rng: np.random.Generator,
    *tensors: Tensor,
) -> tuple[Tensor, ...]:
    """Adjust a group of ``(C, T)`` tensors to exactly ``num_channels`` channels.

    All tensors receive the **same** channel indices so that linear relationships
    (e.g. ``noisy == speech + noise``) are preserved after the adjustment.

    Note:
        When ``c == num_channels`` the original tensors are returned as-is
        (no copy).  Callers must not mutate them in place.
    """
    c = tensors[0].shape[0]

    if not all(t.shape[0] == c for t in tensors[1:]):
        raise ValueError(
            f"all tensors must have the same channel count, "
            f"got {[t.shape[0] for t in tensors]}"
        )

    if c == num_channels:
        return tensors

    if c < num_channels:
        ch_idx = torch.as_tensor(
            rng.integers(0, c, size=num_channels - c), dtype=torch.long
        )
        return tuple(torch.cat([t, t[ch_idx]], dim=0) for t in tensors)

    ch_idx = torch.as_tensor(
        rng.choice(c, num_channels, replace=False), dtype=torch.long
    )
    return tuple(t[ch_idx] for t in tensors)


def fft_convolve(signal: Tensor, kernel: Tensor) -> Tensor:
    """FFT convolution of a ``(C, T)`` signal with a 1-D kernel, trimmed to T."""
    out_len = signal.shape[1]
    n = 1 << (out_len + kernel.shape[0] - 2).bit_length()
    K = torch.fft.rfft(kernel, n=n)  # [n//2+1]
    S = torch.fft.rfft(signal, n=n)  # [C, n//2+1]
    return torch.fft.irfft(S * K, n=n)[:, :out_len]  # [C, out_len]


def compute_stft(audio: Tensor, fft_size: int, hop_size: int, window: Tensor) -> Tensor:
    """STFT consistent with FftDataset (center=False, hann window).

    Args:
        audio (Tensor): Input waveform of shape ``[B, C, T_samples]``.
        fft_size (int): FFT size.
        hop_size (int): Hop size between frames.
        window (Tensor): Analysis window of length ``fft_size``.

    Returns:
        Tensor: Real-valued STFT output of shape ``[B, C, T_frames, F, 2]``.
    """
    B, C, T = audio.shape
    out = torch.stft(
        audio.reshape(B * C, T),
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=fft_size,
        window=window,
        return_complex=True,
        center=False,
    )
    # out: [B*C, F, T_frames] → [B, C, T_frames, F, 2]
    F_bins, T_frames = out.shape[-2], out.shape[-1]
    out = out.view(B, C, F_bins, T_frames).permute(0, 1, 3, 2)
    return torch.view_as_real(out.contiguous())


def spec_to_audio(spec: Tensor, fft_size: int, hop_size: int, window: Tensor) -> Tensor:
    """Reconstruct waveform from a complex spectrogram tensor.

    Args:
        spec: Shape ``[B, 1, T_frames, F, 2]`` real-valued (real/imag).
        fft_size: FFT size.
        hop_size: Hop size.
        window: Analysis/synthesis window of length ``fft_size``.

    Returns:
        Waveform of shape ``[B, T_samples]``.
    """
    # [B, 1, T, F, 2] → complex [B, 1, T, F] → [B, F, T]
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
    )  # [B, T_samples]
    return audio


def spectrogram_to_db(spec: Tensor, ref_db: float = 80.0) -> np.ndarray:
    """Convert a complex spectrogram tensor to a dB magnitude array.

    Args:
        spec: Shape ``[1, T_frames, F, 2]`` or ``[T_frames, F, 2]``.
        ref_db: Dynamic range in dB to clip at.

    Returns:
        Array of shape ``[F, T_frames]`` with values in ``[-ref_db, 0]`` dB.
    """
    if spec.dim() == 4:
        spec = spec[0]  # [T, F, 2]
    mag = torch.view_as_complex(spec.contiguous()).abs()  # [T, F]
    mag_db = 20.0 * torch.log10(mag + 1e-8)
    mag_db = mag_db - mag_db.max()
    mag_db = mag_db.clamp(min=-ref_db)
    return mag_db.T.cpu().numpy()  # [F, T]


def load_audio_mono_resampled(
    audio_path: str, target_sr: int, resample_method: str = "kaiser_best"
) -> tuple[Tensor, int]:
    """Load audio, convert to mono, float32, and resample to ``target_sr``."""
    audio, sr = torchaudio.load(audio_path)
    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(torch.float32)
    if sr != target_sr:
        audio = resample(audio, sr, target_sr, method=resample_method)
        sr = target_sr
    return audio, sr


def save_audio_peak_normalized(
    audio: Tensor,
    out_path: str,
    sample_rate: int,
    eps: float = 1e-8,
) -> None:
    """Peak-normalize mono waveform and save as WAV."""
    wav = audio.detach().cpu().numpy().astype(np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0:
        wav = wav / max(peak, eps)
    out = torch.from_numpy(wav).unsqueeze(0)
    torchaudio.save(out_path, out, sample_rate)
