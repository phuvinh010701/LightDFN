import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


@dataclass
class ModelConfig:
    """Model configuration matching DeepFilterNet3 defaults."""

    # Audio parameters
    sr: int = 48000
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_df: int = 96

    # Normalization
    norm_tau: float = 1.0

    # LSNR parameters
    lsnr_max: int = 35
    lsnr_min: int = -15

    # ERB parameters
    min_nb_freqs: int = 2

    # DF parameters
    df_order: int = 5
    df_lookahead: int = 2
    pad_mode: str = "output"

    # Convolution parameters
    conv_lookahead: int = 2
    conv_ch: int = 64
    conv_depthwise: bool = True
    convt_depthwise: bool = False
    conv_kernel: tuple[int, int] = field(default_factory=lambda: (1, 3))
    convt_kernel: tuple[int, int] = field(default_factory=lambda: (1, 3))
    conv_kernel_inp: tuple[int, int] = field(default_factory=lambda: (3, 3))

    # Embedding/Encoder GRU parameters
    emb_hidden_dim: int = 256
    emb_num_layers: int = 3
    emb_gru_skip: str = "none"
    emb_gru_skip_enc: str = "none"

    # Deep Filtering decoder GRU parameters
    df_hidden_dim: int = 256
    df_num_layers: int = 2
    df_gru_skip: str = "groupedlinear"
    df_pathway_kernel_size_t: int = 5

    # Linear layer parameters
    lin_groups: int = 16
    enc_lin_groups: int = 32

    # Other architecture parameters
    enc_concat: bool = False
    df_n_iter: int = 1

    # Post-processing
    mask_pf: bool = False
    pf_beta: float = 0.02
    lsnr_dropout: bool = False

    # Li-GRU specific — should match DataLoaderConfig.batch_size at training time
    batch_size: int = 1


@dataclass
class AugmentationConfig:
    """Augmentation probabilities for all pipeline stages."""

    # Speech augmentations (get_speech_augmentations)
    p_remove_dc: float = 0.25
    p_lfilt: float = 0.30
    p_biquad: float = 0.30
    p_resample: float = 0.20
    p_clipping: float = 0.10

    # Noise augmentations (get_noise_augmentations)
    p_noise_clipping: float = 0.15
    p_noise_biquad: float = 0.40

    # Speech distortions — time domain (get_speech_distortions_td)
    p_zeroing: float = 0.10
    p_air_absorption: float = 0.05
    p_bandwidth_limit: float = 0.20


@dataclass
class DataLoaderConfig:
    """Configuration for building training/validation dataloaders."""

    speech_hdf5: list[str] = field(default_factory=list)
    noise_hdf5: list[str] = field(default_factory=list)
    rir_hdf5: list[str] = field(default_factory=list)
    sr: int = 48_000
    max_len_s: float = 5.0
    batch_size: int = 4
    num_workers: int = 4
    seed: int = 42
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_spec: int = 96
    # Must match ModelConfig.min_nb_freqs so feature and model ERB filterbanks agree
    min_nb_freqs: int = 2
    # Exponential decay time-constant (seconds) for running feature normalization
    norm_tau: float = 1.0


@dataclass
class TrainConfig:
    """Training loop configuration."""

    epochs: int = 100
    lr: float = 5e-4
    lr_min: float = 1e-5
    warmup_epochs: int = 3
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "lightdfn"
    log_every_n_steps: int = 100
    val_every_n_epochs: int = 1
    max_nans: int = 50


def load_config(
    path: str | os.PathLike | None = None,
) -> tuple[ModelConfig, AugmentationConfig, DataLoaderConfig, TrainConfig]:
    """Load all configs from a YAML file.

    Args:
        path: Path to a YAML config file. Defaults to src/configs/default.yaml.

    Returns:
        Tuple of (ModelConfig, AugmentationConfig, DataLoaderConfig, TrainConfig).
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    augmentation_raw = raw.get("augmentation", {})
    data_loader_raw = raw.get("data_loader", {})
    train_raw = raw.get("train", {})

    for key in ("conv_kernel", "convt_kernel", "conv_kernel_inp"):
        if key in model_raw:
            model_raw[key] = tuple(model_raw[key])

    model_cfg = ModelConfig(**model_raw)
    augmentation_cfg = AugmentationConfig(**augmentation_raw)
    data_loader_cfg = DataLoaderConfig(**data_loader_raw)
    train_cfg = TrainConfig(**train_raw)

    return model_cfg, augmentation_cfg, data_loader_cfg, train_cfg
