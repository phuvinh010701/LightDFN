import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.augmentations import get_speech_augmentations
from src.configs.config import AugmentationConfig
from src.constants import AUDIO_DATASET_TYPES
from src.dataloader.hdf5 import Hdf5Dataset


class TdDataset(Dataset):
    def __init__(
        self,
        speech_hdf5_file: str,
        noise_hdf5_file: str,
        rir_hdf5_file: str,
        augmentation_config: AugmentationConfig,
        sr: int = 48000,
        snrs: list[int] = [-5, 0, 5, 10, 20, 40],
        gains: list[int] = [-6, 0, 6],
        split: AUDIO_DATASET_TYPES = "train",
        seed: int = 42,
    ):
        super().__init__()

        self.speech_dataset = Hdf5Dataset(speech_hdf5_file, sampling_factor=1.0)
        self.noise_dataset = Hdf5Dataset(noise_hdf5_file, sampling_factor=1.0)
        self.rir_dataset = Hdf5Dataset(rir_hdf5_file, sampling_factor=1.0)

        self.snrs = snrs
        self.gains = gains

        self.speech_aug = get_speech_augmentations(
            augmentation_config, sample_rate=sr, seed=seed
        )

    def __getitem__(self, index: int) -> Tensor:
        speech = self.speech_dataset[index]
        speech = torch.from_numpy(speech)

        if speech.ndim == 2:
            speech = speech.unsqueeze(0)

        speech = self.speech_aug(speech)

        speech = speech.squeeze(0)
        return speech


if __name__ == "__main__":
    from src.configs.config import load_config

    model_config, augmentation_config = load_config()

    td_dataset = TdDataset(
        speech_hdf5_file="./datasets/hdf5/speech_clean_minisize.hdf5",
        noise_hdf5_file="./datasets/hdf5/noise_music_minisize.hdf5",
        rir_hdf5_file="./datasets/hdf5/rir_minisize.hdf5",
        augmentation_config=augmentation_config,
        sr=model_config.sr,
    )

    test_speech = td_dataset[0]
    print(test_speech.shape)
