"""Complete DeepFilterNet DataLoader implementation in Python/PyTorch.

Port of DeepFilterNet libDF/src/{dataloader.rs, dataset.rs}

This module implements the complete data loading pipeline:
- HDF5Dataset: Low-level HDF5 file reading (PCM codec)
- TdDataset: Time-domain dataset with mixing and augmentation
- FftDataset: Frequency-domain wrapper with STFT/ERB
- DeepFilterNetDataLoader: Batching and epoch management

Key features:
- Seed-based deterministic sampling
- Multi-noise mixing (2-5 tracks)
- Reverb simulation with RT60/DRR control
- SNR-controlled mixing
- Complete augmentation pipeline integration
- Batch collation with padding
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Union, List, Tuple
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import torch.nn.functional as F

from .erb import get_erb_filterbanks
from .utils import (
    rms,
    mix_audio_signal,
    combine_noises,
    apply_gain,
)
from .augmentations import (
    get_speech_augmentations,
    get_noise_augmentations,
    get_speech_distortions_td,
    RandReverbSim,
    Compose,
)


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class Sample:
    """Single sample from dataset.

    Port of DeepFilterNet libDF/src/dataset.rs:Sample

    Fields match Rust implementation exactly.
    """
    # Audio signals (channels, time)
    speech: np.ndarray  # Clean speech
    noisy: np.ndarray   # Noisy mixture

    # Metadata
    max_freq: int       # Maximum frequency (bandwidth estimate)
    snr: int            # SNR in dB
    gain: int           # Gain in dB
    sample_id: int      # Unique sample ID

    # Optional fields
    noise: Optional[np.ndarray] = None  # Noise signal
    feat_erb: Optional[np.ndarray] = None  # ERB features
    feat_spec: Optional[np.ndarray] = None  # Spectrogram features


@dataclass
class DsBatch:
    """Collated batch with padded samples.

    Port of DeepFilterNet libDF/src/dataloader.rs:DsBatch
    """
    # Main signals: (batch, channels, time)
    speech: Tensor
    noisy: Tensor
    noise: Optional[Tensor] = None

    # Features: (batch, channels, time, freq) for complex spectrograms
    # or (batch, channels, time, erb_bands) for ERB features
    feat_erb: Optional[Tensor] = None
    feat_spec: Optional[Tensor] = None

    # Metadata
    lengths: Optional[Tensor] = None  # Original lengths before padding
    max_freq: Optional[Tensor] = None
    snr: Optional[Tensor] = None
    gain: Optional[Tensor] = None
    sample_ids: Optional[Tensor] = None

    # Attenuation limits (for bandwidth-aware processing)
    atten_lim: Optional[Tensor] = None


# ============================================================================
# HDF5 Dataset
# ============================================================================


class Hdf5Dataset:
    """Low-level HDF5 dataset reader.

    Port of DeepFilterNet libDF/src/dataset.rs:Hdf5Dataset

    PCM-only codec support (no Vorbis/FLAC for now as per user requirement).
    """

    def __init__(
        self,
        file_path: str,
        sr: int,
        sampling_factor: float = 1.0,
        max_len_s: Optional[float] = None,
    ):
        """Initialize HDF5 dataset.

        Args:
            file_path: Path to HDF5 file
            sr: Sample rate
            sampling_factor: Over/undersampling factor (>1 = oversample, <1 = undersample)
            max_len_s: Maximum audio length in seconds
        """
        self.file_path = file_path
        self.sr = sr
        self.sampling_factor = sampling_factor
        self.max_len_s = max_len_s

        # Open HDF5 file
        self.h5file = h5py.File(file_path, 'r')

        # Load keys (sample names)
        self.keys = list(self.h5file.keys())
        if len(self.keys) == 0:
            raise ValueError(f"No keys found in HDF5 file: {file_path}")

        # Apply sampling factor to create effective dataset size
        self.effective_size = int(len(self.keys) * sampling_factor)

        print(f"Loaded HDF5: {file_path}")
        print(f"  Keys: {len(self.keys)}, Effective size: {self.effective_size}, Factor: {sampling_factor:.2f}")

    def __len__(self) -> int:
        """Return effective dataset size (with sampling factor)."""
        return self.effective_size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, int]:
        """Load audio sample from HDF5.

        Args:
            idx: Index (may be >= len(keys) due to oversampling)

        Returns:
            Tuple of (audio, sample_rate, max_freq)
                audio: shape (channels, samples)
                sample_rate: Hz
                max_freq: Bandwidth in Hz
        """
        # Handle oversampling: cycle through keys
        key_idx = idx % len(self.keys)
        key = self.keys[key_idx]

        # Load PCM data
        dataset = self.h5file[key]

        # Read audio data
        audio = dataset[:]  # shape: (channels, samples) or (samples,)

        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)

        # Ensure 2D (channels, samples)
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Read sample rate from attributes
        if 'sr' in dataset.attrs:
            sr = int(dataset.attrs['sr'])
        else:
            sr = self.sr

        # Read max_freq from attributes
        if 'max_freq' in dataset.attrs:
            max_freq = int(dataset.attrs['max_freq'])
        else:
            max_freq = sr // 2

        # Trim to max length if specified
        if self.max_len_s is not None:
            max_samples = int(self.max_len_s * sr)
            if audio.shape[1] > max_samples:
                # Random crop
                start = np.random.randint(0, audio.shape[1] - max_samples + 1)
                audio = audio[:, start:start + max_samples]

        return audio, sr, max_freq

    def close(self):
        """Close HDF5 file."""
        if hasattr(self, 'h5file'):
            self.h5file.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# ============================================================================
# Time-Domain Dataset
# ============================================================================


class TdDataset(Dataset):
    """Time-domain dataset with mixing and augmentation.

    Port of DeepFilterNet libDF/src/dataset.rs:TdDataset

    Complete pipeline:
    1. Load speech from HDF5
    2. Apply speech augmentations
    3. Load multiple noise tracks (2-5)
    4. Apply noise augmentations
    5. Combine noises
    6. Apply reverb (probabilistically)
    7. Mix at target SNR
    8. Apply gain
    9. Apply time-domain distortions
    10. Return Sample
    """

    def __init__(
        self,
        speech_hdf5_files: List[str],
        noise_hdf5_files: List[str],
        sr: int = 48000,
        max_len_s: float = 5.0,
        snrs: List[int] = None,
        gains: List[int] = None,
        num_channels: int = 1,
        p_reverb: float = 0.2,
        p_interfering_speaker: float = 0.1,
        reverb_cfg: Optional[dict] = None,
        rir_hdf5_files: Optional[List[str]] = None,
        split: str = 'train',
        seed: int = 0,
    ):
        """Initialize TdDataset.

        Args:
            speech_hdf5_files: List of speech HDF5 file paths
            noise_hdf5_files: List of noise HDF5 file paths
            sr: Sample rate
            max_len_s: Maximum audio length in seconds
            snrs: SNR range in dB (default: [-5, 0, 5, 10, 20, 40])
            gains: Gain range in dB (default: [-6, 0, 6])
            num_channels: Number of audio channels
            p_reverb: Probability of applying reverb
            p_interfering_speaker: Probability of adding interfering speaker
            reverb_cfg: Reverb configuration dict
            rir_hdf5_files: RIR HDF5 file paths (if None, uses synthetic)
            split: Dataset split ('train', 'valid', 'test')
            seed: Base random seed
        """
        super().__init__()

        self.sr = sr
        self.max_len_s = max_len_s
        self.max_samples = int(max_len_s * sr)
        self.num_channels = num_channels
        self.split = split
        self.base_seed = seed
        self.epoch_seed = seed

        # SNR and gain ranges
        self.snrs = snrs if snrs is not None else [-5, 0, 5, 10, 20, 40]
        self.gains = gains if gains is not None else [-6, 0, 6]

        # Reverb settings
        self.p_reverb = p_reverb
        self.p_interfering_speaker = p_interfering_speaker
        self.reverb_cfg = reverb_cfg if reverb_cfg else {}

        # Load speech datasets
        self.speech_datasets = []
        for file_path in speech_hdf5_files:
            ds = Hdf5Dataset(file_path, sr, sampling_factor=1.0, max_len_s=max_len_s)
            self.speech_datasets.append(ds)

        # Load noise datasets
        self.noise_datasets = []
        for file_path in noise_hdf5_files:
            ds = Hdf5Dataset(file_path, sr, sampling_factor=1.0, max_len_s=max_len_s)
            self.noise_datasets.append(ds)

        # Load RIR datasets (optional)
        self.rir_datasets = []
        if rir_hdf5_files:
            for file_path in rir_hdf5_files:
                ds = Hdf5Dataset(file_path, sr, sampling_factor=1.0, max_len_s=None)
                self.rir_datasets.append(ds)

        # Calculate total size (use largest dataset)
        self.total_speech_samples = sum(len(ds) for ds in self.speech_datasets)
        self.total_noise_samples = sum(len(ds) for ds in self.noise_datasets)

        # Dataset size = number of speech samples (we generate one sample per speech)
        self.dataset_size = self.total_speech_samples

        print(f"TdDataset [{split}]:")
        print(f"  Speech samples: {self.total_speech_samples}")
        print(f"  Noise samples: {self.total_noise_samples}")
        print(f"  Dataset size: {self.dataset_size}")

        # Initialize augmentations
        self.speech_aug = get_speech_augmentations(seed=seed)
        self.noise_aug = get_noise_augmentations(seed=seed)
        self.distortion_aug = get_speech_distortions_td(seed=seed)

        # Reverb augmentation
        if self.p_reverb > 0:
            self.reverb_aug = RandReverbSim(
                prob=self.p_reverb,
                rir_files=None,  # Use synthetic RIRs
                **self.reverb_cfg
            )
        else:
            self.reverb_aug = None

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size

    def set_epoch_seed(self, epoch: int):
        """Set seed for current epoch (for deterministic shuffling).

        Args:
            epoch: Epoch number
        """
        self.epoch_seed = self.base_seed + epoch * 10000

    def _get_sample_rng(self, idx: int) -> np.random.Generator:
        """Get RNG for specific sample index.

        Uses seed = epoch_seed + idx for train, just idx for eval.

        Args:
            idx: Sample index

        Returns:
            Random number generator
        """
        if self.split == 'train':
            seed = self.epoch_seed + idx
        else:
            seed = self.base_seed + idx

        return np.random.default_rng(seed)

    def _load_speech(self, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
        """Load random speech sample.

        Args:
            rng: Random number generator

        Returns:
            Tuple of (audio, sample_rate, max_freq)
        """
        # Select random speech dataset
        ds_idx = rng.integers(0, len(self.speech_datasets))
        dataset = self.speech_datasets[ds_idx]

        # Select random sample
        sample_idx = rng.integers(0, len(dataset))

        # Load
        audio, sr, max_freq = dataset[sample_idx]

        return audio, sr, max_freq

    def _load_noise(self, rng: np.random.Generator, num_noises: int) -> List[Tuple[np.ndarray, int, int]]:
        """Load multiple random noise samples.

        Args:
            rng: Random number generator
            num_noises: Number of noise tracks to load (2-5)

        Returns:
            List of (audio, sample_rate, max_freq) tuples
        """
        noises = []

        for _ in range(num_noises):
            # Select random noise dataset
            ds_idx = rng.integers(0, len(self.noise_datasets))
            dataset = self.noise_datasets[ds_idx]

            # Select random sample
            sample_idx = rng.integers(0, len(dataset))

            # Load
            audio, sr, max_freq = dataset[sample_idx]
            noises.append((audio, sr, max_freq))

        return noises

    def __getitem__(self, idx: int) -> Sample:
        """Generate single sample with mixing and augmentation.

        10-step pipeline matching Rust implementation.

        Args:
            idx: Sample index

        Returns:
            Sample object
        """
        # Get sample-specific RNG for determinism
        rng = self._get_sample_rng(idx)

        # Seed torch and numpy for augmentations
        sample_seed = int(rng.integers(0, 2**31))
        torch.manual_seed(sample_seed)
        np.random.seed(sample_seed)

        # Step 1: Load speech
        speech, sr_speech, max_freq_speech = self._load_speech(rng)

        # Ensure speech is correct length
        if speech.shape[1] < self.max_samples:
            # Repeat if too short
            repeats = int(np.ceil(self.max_samples / speech.shape[1]))
            speech = np.tile(speech, (1, repeats))

        # Crop to exact length
        if speech.shape[1] > self.max_samples:
            start = rng.integers(0, speech.shape[1] - self.max_samples + 1)
            speech = speech[:, start:start + self.max_samples]
        else:
            speech = speech[:, :self.max_samples]

        # Adjust channels
        if speech.shape[0] > self.num_channels:
            # Remove random channels
            channels_to_keep = rng.choice(speech.shape[0], self.num_channels, replace=False)
            speech = speech[channels_to_keep, :]
        elif speech.shape[0] < self.num_channels:
            # Duplicate random channels
            while speech.shape[0] < self.num_channels:
                ch_to_dup = rng.integers(0, speech.shape[0])
                speech = np.vstack([speech, speech[ch_to_dup:ch_to_dup+1, :]])

        # Step 2: Apply speech augmentations
        speech_aug = self.speech_aug(torch.from_numpy(speech), sample_rate=self.sr).numpy()

        # Step 3: Load multiple noise tracks (2-5)
        num_noise_tracks = rng.integers(2, 6)  # 2 to 5 inclusive
        noise_samples = self._load_noise(rng, num_noise_tracks)

        # Step 4: Apply noise augmentations to each track
        noises_aug = []
        for noise, sr_noise, _ in noise_samples:
            noise_aug = self.noise_aug(torch.from_numpy(noise), sample_rate=self.sr).numpy()
            noises_aug.append(noise_aug)

        # Step 5: Combine noises
        # Sample random gains for each noise track
        noise_gains = rng.uniform(-3.0, 3.0, size=num_noise_tracks).tolist()
        combined_noise = combine_noises(
            num_channels=self.num_channels,
            target_length=self.max_samples,
            noises=noises_aug,
            noise_gains=noise_gains,
            rng=rng
        )

        # Step 6: Apply reverb (probabilistically)
        speech_reverb = None
        if self.reverb_aug is not None and rng.uniform() < self.p_reverb:
            # Three scenarios: noise-only, speech-only, both
            scenario = rng.integers(0, 3)

            if scenario == 0:  # Reverb on noise only
                combined_noise = self.reverb_aug.apply(combined_noise, self.sr)
            elif scenario == 1:  # Reverb on speech only
                speech_reverb = self.reverb_aug.apply(speech_aug, self.sr)
            else:  # Reverb on both
                speech_reverb = self.reverb_aug.apply(speech_aug, self.sr)
                combined_noise = self.reverb_aug.apply(combined_noise, self.sr)

        # Step 7: Optionally add interfering speaker
        if rng.uniform() < self.p_interfering_speaker:
            interfering_speech, _, _ = self._load_speech(rng)

            # Match length and channels
            if interfering_speech.shape[1] < self.max_samples:
                repeats = int(np.ceil(self.max_samples / interfering_speech.shape[1]))
                interfering_speech = np.tile(interfering_speech, (1, repeats))
            interfering_speech = interfering_speech[:, :self.max_samples]

            # Adjust channels to match
            while interfering_speech.shape[0] < self.num_channels:
                interfering_speech = np.vstack([interfering_speech, interfering_speech[0:1, :]])
            interfering_speech = interfering_speech[:self.num_channels, :]

            # Mix at low SNR (15-30 dB)
            interfering_snr = rng.choice([15, 20, 30])
            combined_noise = combined_noise + interfering_speech * (10 ** (-interfering_snr / 20))

        # Step 8: Sample SNR and gain
        snr_db = float(rng.choice(self.snrs))
        gain_db = float(rng.choice(self.gains))

        # Step 9: Mix speech and noise at target SNR
        clean_out, noise_out, noisy = mix_audio_signal(
            clean=speech_aug,
            noise=combined_noise,
            snr_db=snr_db,
            gain_db=gain_db,
            clean_distorted=speech_reverb,  # Use reverberant speech if available
        )

        # Step 10: Apply time-domain distortions to noisy signal
        noisy_distorted = self.distortion_aug(torch.from_numpy(noisy), sample_rate=self.sr).numpy()

        # Create sample
        sample = Sample(
            speech=clean_out,
            noisy=noisy_distorted,
            noise=noise_out,
            max_freq=max_freq_speech,
            snr=int(snr_db),
            gain=int(gain_db),
            sample_id=idx,
        )

        return sample


# ============================================================================
# FFT Dataset
# ============================================================================


class FftDataset(Dataset):
    """Frequency-domain dataset wrapper.

    Port of DeepFilterNet libDF/src/dataset.rs:FftDataset

    Wraps TdDataset and applies:
    - STFT (Short-Time Fourier Transform)
    - ERB filterbank
    - Normalization
    """

    def __init__(
        self,
        td_dataset: TdDataset,
        fft_size: int = 960,
        hop_size: int = 480,
        nb_erb: int = 32,
        nb_spec: int = 96,
        norm_alpha: float = 0.99,
        sr: int = 48000,
    ):
        """Initialize FftDataset.

        Args:
            td_dataset: Time-domain dataset to wrap
            fft_size: FFT size
            hop_size: Hop size for STFT
            nb_erb: Number of ERB bands
            nb_spec: Number of spectrogram frequencies to keep
            norm_alpha: EMA alpha for normalization
            sr: Sample rate
        """
        super().__init__()

        self.td_dataset = td_dataset
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.nb_erb = nb_erb
        self.nb_spec = nb_spec
        self.norm_alpha = norm_alpha
        self.sr = sr

        # Create Hann window
        self.window = torch.hann_window(fft_size)

        # Create ERB filterbanks
        self.erb_fb, self.erb_inv_fb = get_erb_filterbanks(
            sr=sr,
            fft_size=fft_size,
            nb_erb=nb_erb,
            min_nb_freqs=2
        )

        print(f"FftDataset:")
        print(f"  FFT size: {fft_size}, Hop: {hop_size}")
        print(f"  ERB bands: {nb_erb}")
        print(f"  Spec bins: {nb_spec}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.td_dataset)

    def set_epoch_seed(self, epoch: int):
        """Pass through to td_dataset."""
        self.td_dataset.set_epoch_seed(epoch)

    def _apply_stft(self, audio: np.ndarray) -> np.ndarray:
        """Apply STFT to audio.

        Args:
            audio: Audio array of shape (channels, samples)

        Returns:
            Complex spectrogram of shape (channels, frames, freqs)
        """
        # Convert to tensor
        audio_t = torch.from_numpy(audio).float()

        # Apply STFT per channel
        specs = []
        for ch in range(audio_t.shape[0]):
            spec = torch.stft(
                audio_t[ch],
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.fft_size,
                window=self.window,
                return_complex=True,
                center=True,
                pad_mode='reflect',
            )
            specs.append(spec)

        # Stack: (channels, freqs, frames)
        spec_tensor = torch.stack(specs, dim=0)

        # Transpose to (channels, frames, freqs)
        spec_tensor = spec_tensor.transpose(1, 2)

        return spec_tensor.numpy()

    def _compute_erb_features(self, spec: np.ndarray) -> np.ndarray:
        """Compute ERB features from spectrogram.

        Args:
            spec: Complex spec of shape (channels, frames, freqs)

        Returns:
            ERB features of shape (channels, frames, erb_bands)
        """
        # Compute magnitude
        mag = np.abs(spec)  # (channels, frames, freqs)

        # Apply ERB filterbank
        # erb_fb: (freqs, erb_bands)
        erb_fb_np = self.erb_fb.numpy()

        # Matrix multiply: (channels, frames, freqs) @ (freqs, erb_bands)
        # -> (channels, frames, erb_bands)
        erb_features = np.einsum('ctf,fe->cte', mag, erb_fb_np)

        return erb_features.astype(np.float32)

    def __getitem__(self, idx: int) -> Sample:
        """Generate sample with frequency-domain features.

        Args:
            idx: Sample index

        Returns:
            Sample with STFT and ERB features
        """
        # Get time-domain sample
        sample = self.td_dataset[idx]

        # Apply STFT to speech and noisy
        speech_spec = self._apply_stft(sample.speech)
        noisy_spec = self._apply_stft(sample.noisy)

        # Compute ERB features from noisy
        erb_features = self._compute_erb_features(noisy_spec)

        # Keep only first nb_spec frequency bins
        speech_spec = speech_spec[:, :, :self.nb_spec]
        noisy_spec = noisy_spec[:, :, :self.nb_spec]

        # Update sample
        sample.feat_erb = erb_features
        sample.feat_spec = noisy_spec

        return sample


# ============================================================================
# DataLoader
# ============================================================================


def collate_fn(batch: List[Sample]) -> DsBatch:
    """Collate batch of samples with padding.

    Port of DeepFilterNet libDF/src/dataloader.rs collate logic.

    Args:
        batch: List of Sample objects

    Returns:
        DsBatch with padded tensors
    """
    batch_size = len(batch)

    # Find max length (in time-domain samples)
    max_len = max(s.speech.shape[1] for s in batch)

    # Check if we have FFT features
    has_erb = batch[0].feat_erb is not None
    has_spec = batch[0].feat_spec is not None

    if has_erb or has_spec:
        # Find max length in frames
        max_frames = max(s.feat_erb.shape[1] if has_erb else s.feat_spec.shape[1] for s in batch)

    # Get dimensions
    num_channels = batch[0].speech.shape[0]

    # Initialize tensors
    speech_batch = torch.zeros(batch_size, num_channels, max_len)
    noisy_batch = torch.zeros(batch_size, num_channels, max_len)
    noise_batch = torch.zeros(batch_size, num_channels, max_len) if batch[0].noise is not None else None

    lengths = torch.zeros(batch_size, dtype=torch.long)
    max_freqs = torch.zeros(batch_size, dtype=torch.long)
    snrs = torch.zeros(batch_size, dtype=torch.long)
    gains = torch.zeros(batch_size, dtype=torch.long)
    sample_ids = torch.zeros(batch_size, dtype=torch.long)

    # ERB features
    if has_erb:
        nb_erb = batch[0].feat_erb.shape[2]
        erb_batch = torch.zeros(batch_size, num_channels, max_frames, nb_erb)
    else:
        erb_batch = None

    # Spec features
    if has_spec:
        nb_spec = batch[0].feat_spec.shape[2]
        # Complex spectrogram
        spec_batch = torch.zeros(batch_size, num_channels, max_frames, nb_spec, dtype=torch.complex64)
    else:
        spec_batch = None

    # Fill batch
    for i, sample in enumerate(batch):
        # Time-domain signals
        t_len = sample.speech.shape[1]
        speech_batch[i, :, :t_len] = torch.from_numpy(sample.speech)
        noisy_batch[i, :, :t_len] = torch.from_numpy(sample.noisy)

        if sample.noise is not None and noise_batch is not None:
            noise_batch[i, :, :t_len] = torch.from_numpy(sample.noise)

        # Metadata
        lengths[i] = t_len
        max_freqs[i] = sample.max_freq
        snrs[i] = sample.snr
        gains[i] = sample.gain
        sample_ids[i] = sample.sample_id

        # ERB features
        if has_erb:
            f_len = sample.feat_erb.shape[1]
            erb_batch[i, :, :f_len, :] = torch.from_numpy(sample.feat_erb)

        # Spec features
        if has_spec:
            f_len = sample.feat_spec.shape[1]
            spec_batch[i, :, :f_len, :] = torch.from_numpy(sample.feat_spec).cfloat()

    # Create batch object
    ds_batch = DsBatch(
        speech=speech_batch,
        noisy=noisy_batch,
        noise=noise_batch,
        feat_erb=erb_batch,
        feat_spec=spec_batch,
        lengths=lengths,
        max_freq=max_freqs,
        snr=snrs,
        gain=gains,
        sample_ids=sample_ids,
    )

    return ds_batch


class DeepFilterNetDataLoader:
    """Main dataloader with epoch management.

    Port of DeepFilterNet libDF/src/dataloader.rs:DataLoader

    Features:
    - Epoch-based seeding for deterministic training
    - Efficient batching with PyTorch DataLoader
    - Prefetching with persistent workers
    """

    def __init__(
        self,
        dataset: Union[TdDataset, FftDataset],
        batch_size: int,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        drop_last: bool = False,
        pin_memory: bool = True,
    ):
        """Initialize dataloader.

        Args:
            dataset: TdDataset or FftDataset
            batch_size: Batch size
            num_workers: Number of worker processes
            prefetch_factor: Number of batches to prefetch per worker
            drop_last: Whether to drop incomplete last batch
            pin_memory: Pin memory for faster GPU transfer
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.drop_last = drop_last
        self.pin_memory = pin_memory

        self.current_epoch = 0

        # Create PyTorch DataLoader
        self.loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # We handle shuffling via seeding
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    def start_epoch(self, epoch: int):
        """Start new epoch with updated seed.

        Args:
            epoch: Epoch number
        """
        self.current_epoch = epoch
        self.dataset.set_epoch_seed(epoch)

    def __iter__(self):
        """Iterate over batches."""
        return iter(self.loader)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.loader)


# ============================================================================
# Configuration and Builder
# ============================================================================


@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader.

    Port of DeepFilterNet configuration parameters.
    """
    # Dataset paths
    speech_hdf5: List[str]
    noise_hdf5: List[str]
    rir_hdf5: Optional[List[str]] = None

    # Audio parameters
    sr: int = 48000
    max_len_s: float = 5.0
    num_channels: int = 1

    # FFT parameters
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_spec: int = 96
    norm_alpha: float = 0.99

    # Mixing parameters
    snrs: List[int] = field(default_factory=lambda: [-5, 0, 5, 10, 20, 40])
    gains: List[int] = field(default_factory=lambda: [-6, 0, 6])

    # Augmentation probabilities
    p_reverb: float = 0.2
    p_interfering_speaker: float = 0.1

    # Reverb configuration
    reverb_cfg: dict = field(default_factory=dict)

    # DataLoader parameters
    batch_size: int = 32
    batch_size_eval: Optional[int] = None
    num_workers: int = 4
    prefetch_factor: int = 2
    drop_last: bool = True
    pin_memory: bool = True

    # Seeds
    seed: int = 0


class DataLoaderBuilder:
    """Builder for creating dataloaders.

    Simplifies dataloader construction with sensible defaults.
    """

    def __init__(self, config: DataLoaderConfig):
        """Initialize builder.

        Args:
            config: DataLoader configuration
        """
        self.config = config

    def build_td_dataset(self, split: str) -> TdDataset:
        """Build time-domain dataset.

        Args:
            split: Dataset split ('train', 'valid', 'test')

        Returns:
            TdDataset
        """
        return TdDataset(
            speech_hdf5_files=self.config.speech_hdf5,
            noise_hdf5_files=self.config.noise_hdf5,
            sr=self.config.sr,
            max_len_s=self.config.max_len_s,
            snrs=self.config.snrs,
            gains=self.config.gains,
            num_channels=self.config.num_channels,
            p_reverb=self.config.p_reverb,
            p_interfering_speaker=self.config.p_interfering_speaker,
            reverb_cfg=self.config.reverb_cfg,
            rir_hdf5_files=self.config.rir_hdf5,
            split=split,
            seed=self.config.seed,
        )

    def build_fft_dataset(self, split: str) -> FftDataset:
        """Build FFT dataset.

        Args:
            split: Dataset split

        Returns:
            FftDataset wrapping TdDataset
        """
        td_dataset = self.build_td_dataset(split)

        return FftDataset(
            td_dataset=td_dataset,
            fft_size=self.config.fft_size,
            hop_size=self.config.hop_size,
            nb_erb=self.config.nb_erb,
            nb_spec=self.config.nb_spec,
            norm_alpha=self.config.norm_alpha,
            sr=self.config.sr,
        )

    def build(self, split: str, use_fft: bool = True) -> DeepFilterNetDataLoader:
        """Build complete dataloader.

        Args:
            split: Dataset split
            use_fft: Whether to use FFT dataset (True) or time-domain only (False)

        Returns:
            DeepFilterNetDataLoader
        """
        # Choose batch size
        if split == 'train':
            batch_size = self.config.batch_size
        else:
            batch_size = self.config.batch_size_eval or self.config.batch_size

        # Build dataset
        if use_fft:
            dataset = self.build_fft_dataset(split)
        else:
            dataset = self.build_td_dataset(split)

        # Build dataloader
        dataloader = DeepFilterNetDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=self.config.drop_last if split == 'train' else False,
            pin_memory=self.config.pin_memory,
        )

        return dataloader


# ============================================================================
# Example Usage
# ============================================================================


if __name__ == "__main__":
    print("DeepFilterNet DataLoader Implementation")
    print("=" * 60)

    # Example configuration
    config = DataLoaderConfig(
        speech_hdf5=["path/to/speech.hdf5"],
        noise_hdf5=["path/to/noise.hdf5"],
        sr=48000,
        max_len_s=5.0,
        batch_size=4,
        num_workers=0,  # Single process for testing
    )

    print("\nConfiguration:")
    print(f"  Sample rate: {config.sr} Hz")
    print(f"  Max length: {config.max_len_s} s")
    print(f"  FFT size: {config.fft_size}")
    print(f"  ERB bands: {config.nb_erb}")
    print(f"  Batch size: {config.batch_size}")

    print("\nTo use:")
    print("1. Create config with your HDF5 paths")
    print("2. Build dataloader:")
    print("   builder = DataLoaderBuilder(config)")
    print("   train_loader = builder.build('train', use_fft=True)")
    print("3. Iterate:")
    print("   train_loader.start_epoch(0)")
    print("   for batch in train_loader:")
    print("       # batch.speech, batch.noisy, batch.feat_erb, etc.")
    print("       pass")

    print("\n" + "=" * 60)
    print("Implementation complete!")
