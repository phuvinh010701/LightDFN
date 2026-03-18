import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

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
    conv_kernel: Tuple[int, int] = field(default_factory=lambda: (1, 3))
    convt_kernel: Tuple[int, int] = field(default_factory=lambda: (1, 3))
    conv_kernel_inp: Tuple[int, int] = field(default_factory=lambda: (3, 3))

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

    # Li-GRU specific
    batch_size: int = 1


@dataclass
class AugmentationConfig:
    """Augmentation probabilities."""

    p_remove_dc: float = 0.25
    p_lfilt: float = 0.30
    p_biquad: float = 0.30
    p_resample: float = 0.20
    p_clipping: float = 0.10
    p_noise_clipping: float = 0.15
    p_noise_distortion: float = 0.10


def load_config(
    path: str | os.PathLike | None = None,
) -> tuple[ModelConfig, AugmentationConfig]:
    """Load ModelConfig and AugmentationConfig from a YAML file.

    Args:
        path: Path to a YAML config file. Defaults to config/default.yaml.

    Returns:
        Tuple of (ModelConfig, AugmentationConfig).
    """
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    augmentation_raw = raw.get("augmentation", {})

    for key in ("conv_kernel", "convt_kernel", "conv_kernel_inp"):
        if key in model_raw:
            model_raw[key] = tuple(model_raw[key])

    model_cfg = ModelConfig(**model_raw)
    augmentation_cfg = AugmentationConfig(**augmentation_raw)

    return model_cfg, augmentation_cfg
