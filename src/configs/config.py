import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


@dataclass
class ModelConfig:
    """Model configuration matching DeepFilterNet3 defaults."""

    # [df]
    sr: int = 48000
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_df: int = 96
    norm_tau: float = 1.0
    lsnr_max: int = 35
    lsnr_min: int = -15
    min_nb_freqs: int = 2
    df_order: int = 5
    df_lookahead: int = 2
    pad_mode: str = "output"

    # [deepfilternet]
    conv_lookahead: int = 2
    conv_ch: int = 64
    conv_depthwise: bool = True
    convt_depthwise: bool = False
    conv_kernel: tuple[int, int] = field(default_factory=lambda: (1, 3))
    convt_kernel: tuple[int, int] = field(default_factory=lambda: (1, 3))
    conv_kernel_inp: tuple[int, int] = field(default_factory=lambda: (3, 3))
    emb_hidden_dim: int = 256
    emb_num_layers: int = 3
    emb_gru_skip: str = "none"
    emb_gru_skip_enc: str = "none"
    df_hidden_dim: int = 256
    df_num_layers: int = 2
    df_gru_skip: str = "groupedlinear"
    df_pathway_kernel_size_t: int = 5
    lin_groups: int = 16
    enc_lin_groups: int = 32
    enc_concat: bool = False
    df_n_iter: int = 1
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

    # Noise augmentations (get_noise_augmentations)
    p_noise_clipping: float = 0.15
    p_noise_biquad: float = 0.40

    # [distortion] — speech distortions (time domain + mixing)
    p_reverb: float = 0.1
    p_bandwidth_ext: float = 0.0
    p_clipping: float = 0.0
    p_air_absorption: float = 0.0
    p_zeroing: float = 0.0
    p_interfer_sp: float = 0.0


@dataclass
class DataLoaderConfig:
    """Configuration for building training/validation dataloaders."""

    speech_hdf5: list[str] = field(default_factory=list)
    noise_hdf5: list[str] = field(default_factory=list)
    rir_hdf5: list[str] = field(default_factory=list)

    # [df] audio params (must match ModelConfig)
    sr: int = 48_000
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_spec: int = 96
    min_nb_freqs: int = 2
    norm_tau: float = 1.0

    # [train] data loading params
    max_len_s: float = 3.0
    batch_size: int = 64
    batch_size_eval: int = 64
    num_workers: int = 16
    num_prefetch_batches: int = 8
    seed: int = 43
    dataloader_snrs: list[int] = field(
        default_factory=lambda: [-100, -5, 0, 5, 10, 20, 40]
    )
    global_ds_sampling_f: float = 1.0


@dataclass
class TrainConfig:
    """Training loop configuration."""

    # [train]
    epochs: int = 120
    log_every_n_steps: int = 100
    val_every_n_epochs: int = 1
    early_stopping_patience: int = 25
    max_nans: int = 50
    p_atten_lim: float = 0.0
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "lightdfn"

    # [optim]
    optimizer: str = "adamw"
    lr: float = 1e-3
    lr_warmup: float = 1e-4
    lr_min: float = 1e-6
    warmup_epochs: int = 3
    weight_decay: float = 1e-12
    weight_decay_end: float = 0.01
    adam_betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    amsgrad: bool = True
    grad_clip: float = 1.0
    audio_eval_every_n_epochs: int = 5
    audio_eval_num_samples: int = 3


@dataclass
class LossConfig:
    """Configuration for the combined training loss."""

    # [localsnrloss]
    lsnr_factor: float = 1e-3

    # [maskloss]
    ml_factor: float = 0.0
    ml_mask: str = "iam"
    ml_gamma: float = 0.3
    ml_gamma_pred: float = 0.3
    ml_f_under: float = 1.0
    ml_powers: list[int] = field(default_factory=lambda: [2, 4])
    ml_factors: list[float] = field(default_factory=lambda: [1.0, 10.0])

    # [spectralloss]
    sl_factor_magnitude: float = 0.0
    sl_factor_complex: float = 0.0
    sl_factor_under: float = 1.0
    sl_gamma: float = 0.3

    # [multiresspecloss]
    mrsl_factor: float = 500.0
    mrsl_factor_complex: float = 500.0
    mrsl_gamma: float = 0.3
    mrsl_fft_sizes: list[int] = field(default_factory=lambda: [256, 512, 1024, 2048])

    # [sdrloss]
    sdr_factor: float = 0.0
    sdr_segmental_ws: list[int] = field(default_factory=list)


def load_config(
    path: str | os.PathLike | None = None,
) -> tuple[ModelConfig, AugmentationConfig, DataLoaderConfig, TrainConfig, LossConfig]:
    """Load all configs from a YAML file.

    Args:
        path: Path to a YAML config file. Defaults to src/configs/default.yaml.

    Returns:
        Tuple of (ModelConfig, AugmentationConfig, DataLoaderConfig, TrainConfig, LossConfig).
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    augmentation_raw = raw.get("augmentation", {})
    data_loader_raw = raw.get("data_loader", {})
    train_raw = raw.get("train", {})
    loss_raw = raw.get("loss", {})

    for key in ("conv_kernel", "convt_kernel", "conv_kernel_inp"):
        if key in model_raw:
            model_raw[key] = tuple(model_raw[key])

    model_cfg = ModelConfig(**model_raw)
    augmentation_cfg = AugmentationConfig(**augmentation_raw)
    data_loader_cfg = DataLoaderConfig(**data_loader_raw)
    train_cfg = TrainConfig(**train_raw)
    loss_cfg = LossConfig(**loss_raw)

    return model_cfg, augmentation_cfg, data_loader_cfg, train_cfg, loss_cfg
