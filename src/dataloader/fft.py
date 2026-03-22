"""FFT dataset: wraps TdDataset and adds frequency-domain features."""

import math

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.dataloader.td import Sample, TdDataset
from src.utils.erb import create_erb_fb, erb_fb_widths


def _norm_alpha(sr: int, hop_size: int, norm_tau: float) -> float:
    """Compute exponential smoothing factor alpha from a time-constant tau.

    Matches DeepFilterNet's ``get_norm_alpha``:
    ``alpha = exp(-1 / (frames_per_second * tau))``.

    Args:
        sr: Sample rate in Hz.
        hop_size: STFT hop length in samples.
        norm_tau: Decay time-constant in seconds.

    Returns:
        Smoothing factor alpha in ``[0, 1)``.
    """
    frames_per_sec = sr / hop_size
    return math.exp(-1.0 / (frames_per_sec * norm_tau))


def _running_mean_norm_erb(power_db: Tensor, alpha: float) -> Tensor:
    """Exponential running-mean normalisation over the time axis for ERB features.

    Ports Rust ``band_mean_norm_erb``:
    ``state = x*(1-alpha) + state*alpha;  out = (x - state) / 40``

    State is initialised from -60 dB (band 0) to -90 dB (band E-1), matching
    ``MEAN_NORM_INIT = [-60., -90.]`` in the Rust library.

    Args:
        power_db: Log-power ERB features, shape ``[C, T, E]``.
        alpha: Smoothing factor from :func:`_norm_alpha`.

    Returns:
        Normalised features of shape ``[C, T, E]``, roughly zero-centred.
    """
    C, T, E = power_db.shape
    # Initialise state: linearly from -60 dB (low ERB) to -90 dB (high ERB)
    state = torch.linspace(
        -60.0, -90.0, E, device=power_db.device, dtype=power_db.dtype
    )
    state = state.unsqueeze(0).expand(C, -1).clone()  # [C, E]

    out = torch.empty_like(power_db)
    for t in range(T):
        state = power_db[:, t, :] * (1.0 - alpha) + state * alpha
        out[:, t, :] = (power_db[:, t, :] - state) / 40.0
    return out


def _running_unit_norm(spec: Tensor, alpha: float) -> Tensor:
    """Exponential running unit-normalisation over the time axis for complex features.

    Ports Rust ``band_unit_norm``:
    ``state = |x|*(1-alpha) + state*alpha;  out = x / sqrt(state)``

    State is initialised from 0.001 (bin 0) to 0.0001 (bin F-1), matching
    ``UNIT_NORM_INIT = [0.001, 0.0001]`` in the Rust library.

    Args:
        spec: Complex spectrogram slice, shape ``[C, F, T]``.
        alpha: Smoothing factor from :func:`_norm_alpha`.

    Returns:
        Unit-normalised complex spectrogram of shape ``[C, F, T]``.
    """
    C, F, T = spec.shape
    # Initialise state: linearly from 0.001 (DC) to 0.0001 (highest DF bin)
    state = torch.linspace(0.001, 0.0001, F, device=spec.device, dtype=spec.real.dtype)
    state = state.unsqueeze(0).expand(C, -1).clone()  # [C, F]

    out = torch.empty_like(spec)
    for t in range(T):
        frame = spec[:, :, t]
        state = frame.abs() * (1.0 - alpha) + state * alpha
        out[:, :, t] = frame / state.sqrt()
    return out


class FftDataset(Dataset):
    """Wraps a :class:`TdDataset` and appends per-sample STFT features.

    For each sample the noisy signal is transformed via STFT and two feature
    tensors are computed, matching the normalisation used by the Rust
    ``libDF`` library:

    * ``feat_erb``  — mean log-power per ERB band, running-mean normalised and
      scaled by 1/40, shape ``(channels, frames, nb_erb)``.
    * ``feat_spec`` — first ``nb_spec`` complex STFT bins after per-frequency
      running unit-norm, shape ``(channels, frames, nb_spec)``.

    All other :class:`Sample` fields are passed through unchanged.

    Args:
        td_dataset: Underlying time-domain dataset.
        fft_size:   STFT window / FFT length in samples.
        hop_size:   STFT hop length in samples.
        nb_erb:     Number of ERB filterbank bands.
        nb_spec:    Number of complex STFT bins to keep for ``feat_spec``.
        sr:         Sample rate in Hz (must match ``td_dataset``).
        min_nb_freqs: Minimum frequency bins per ERB band.  Must equal
            ``ModelConfig.min_nb_freqs`` so the feature and model ERB
            filterbanks are identical.
        norm_tau:   Running-mean decay time-constant in seconds.
    """

    def __init__(
        self,
        td_dataset: TdDataset,
        fft_size: int = 960,
        hop_size: int = 480,
        nb_erb: int = 32,
        nb_spec: int = 96,
        sr: int = 48_000,
        min_nb_freqs: int = 2,
        norm_tau: float = 1.0,
    ) -> None:
        self.td_dataset = td_dataset
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.nb_spec = nb_spec
        self.alpha = _norm_alpha(sr, hop_size, norm_tau)

        widths = erb_fb_widths(sr, fft_size, nb_erb, min_nb_freqs=min_nb_freqs)
        self.erb_fb: Tensor = create_erb_fb(widths, sr, normalized=True)  # [F, nb_erb]
        self._window: Tensor = torch.hann_window(fft_size)

    def __len__(self) -> int:
        return len(self.td_dataset)

    def __getitem__(self, idx: int) -> Sample:
        sample = self.td_dataset[idx]

        # STFT of noisy signal: (C, T) → (C, F, frames) complex
        stft = torch.stft(
            sample.noisy,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.fft_size,
            window=self._window,
            return_complex=True,
            center=False,
        )
        # stft: [C, F, T_frames]

        # --- ERB features ---
        # 1. Mean power per ERB band: [C, T, F] @ [F, E] → [C, T, E]
        erb_fb = self.erb_fb.to(stft.device)
        power = stft.abs().pow(2)  # [C, F, T]
        feat_erb = power.permute(0, 2, 1) @ erb_fb  # [C, T, E]

        # 2. Convert to log-power in dB
        feat_erb_db = (feat_erb + 1e-10).log10() * 10.0  # [C, T, E]

        # 3. Running exponential mean normalisation + /40  (matches Rust band_mean_norm_erb)
        feat_erb = _running_mean_norm_erb(feat_erb_db, self.alpha)  # [C, T, E]

        # --- Complex spec features ---
        # Running per-frequency unit normalisation  (matches Rust band_unit_norm)
        spec_slice = stft[:, : self.nb_spec, :]  # [C, F', T]
        spec_normed = _running_unit_norm(spec_slice, self.alpha)  # [C, F', T]
        feat_spec = spec_normed.permute(0, 2, 1)  # [C, T, F']

        return Sample(
            speech=sample.speech,
            noisy=sample.noisy,
            noise=sample.noise,
            snr=sample.snr,
            gain=sample.gain,
            max_freq=sample.max_freq,
            sample_id=sample.sample_id,
            feat_erb=feat_erb,
            feat_spec=feat_spec,
        )
