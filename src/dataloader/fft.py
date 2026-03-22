"""FFT dataset: wraps TdDataset and adds frequency-domain features."""

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.dataloader.td import Sample, TdDataset
from src.utils.erb import create_erb_fb, erb_fb_widths


class FftDataset(Dataset):
    """Wraps a :class:`TdDataset` and appends per-sample STFT features.

    For each sample, the noisy signal is transformed via STFT and two feature
    tensors are computed:

    * ``feat_erb``  — magnitude spectrum projected onto ERB filterbank bands,
      shape ``(channels, frames, nb_erb)``.
    * ``feat_spec`` — first ``nb_spec`` complex STFT bins,
      shape ``(channels, frames, nb_spec)``.

    All other :class:`Sample` fields are passed through unchanged.

    Args:
        td_dataset: Underlying time-domain dataset.
        fft_size:   STFT window / FFT length in samples.
        hop_size:   STFT hop length in samples.
        nb_erb:     Number of ERB filterbank bands.
        nb_spec:    Number of complex STFT bins to keep for ``feat_spec``.
        sr:         Sample rate in Hz (must match ``td_dataset``).
    """

    def __init__(
        self,
        td_dataset: TdDataset,
        fft_size: int = 960,
        hop_size: int = 480,
        nb_erb: int = 32,
        nb_spec: int = 96,
        sr: int = 48_000,
    ) -> None:
        self.td_dataset = td_dataset
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.nb_spec = nb_spec

        widths = erb_fb_widths(sr, fft_size, nb_erb, min_nb_freqs=1)
        self.erb_fb: Tensor = create_erb_fb(widths, sr, normalized=True)  # (F, nb_erb)
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

        # ERB features: (C, F, frames) → (C, frames, nb_erb)
        erb_fb = self.erb_fb.to(stft.device)
        feat_erb = stft.abs().permute(0, 2, 1) @ erb_fb

        # Spec features: first nb_spec bins → (C, frames, nb_spec)
        feat_spec = stft[:, : self.nb_spec, :].permute(0, 2, 1)

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
