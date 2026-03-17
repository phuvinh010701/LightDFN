from torch_audiomentations import Compose, AddColoredNoise


def get_noise_generator(
    p: float = 0.5,
    f_decay_min: float = -2.0,
    f_decay_max: float = 2.0,
    sample_rate: int = 48000,
    output_type: str = "dict",
) -> Compose:
    return Compose(
        [
            AddColoredNoise(
                p=p,
                min_f_decay=f_decay_min,
                max_f_decay=f_decay_max,
                sample_rate=sample_rate,
                output_type=output_type,
            ),
        ]
    )


def get_speech_augmentations() -> Compose:
    return Compose([])


def get_noise_augmentations() -> Compose:
    return Compose([])


def get_speech_distortions_td() -> Compose:
    return Compose([])


if __name__ == "__main__":
    import torch

    augmentations = get_noise_generator(
        p=1.0, f_decay_min=-2.0, f_decay_max=2.0, sample_rate=48000, output_type="dict"
    )
    audio = torch.randn((2, 2, 1000), dtype=torch.float32)