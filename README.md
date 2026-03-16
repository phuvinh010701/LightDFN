# LightDeepFilterNet (LightDFN)

A lightweight, efficient audio noise suppression model that replaces the standard GRU layers in [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) with [Li-GRU](https://github.com/speechbrain/speechbrain) — a computationally cheaper recurrent unit — while preserving the deep filtering architecture.

## Overview

LightDFN targets real-time, resource-constrained deployment scenarios. It maintains the dual-path architecture of DeepFilterNet3 (ERB-based coarse suppression + complex deep filtering) but swaps in Li-GRU cells in the encoder and decoder pathways to reduce parameter count and inference cost.

**Key specs:**

| Parameter | Value |
|---|---|
| Sample rate | 48 kHz |
| FFT / Hop size | 960 / 480 samples |
| ERB bands | 32 |
| DF bins | 96 |
| DF order | 5 |
| Conv channels | 64 |
| Embedding hidden dim | 256 |

## Installation

Requires Python ≥ 3.14 and a CUDA 12.6-compatible GPU (for the default PyTorch index).

```bash
# Clone the repository
git clone https://github.com/phuvinh010701/LightDFN.git
cd LightDFN

# Install dependencies using uv
uv sync
```

## Dataset Setup

LightDFN uses 6 corpora covering clean speech, noise, music, and room impulse responses (RIRs). An automated shell script handles downloading, extraction, license filtering, and manifest generation.

### Datasets

| Dataset | Type | Profile |
|---|---|---|
| VCTK | Clean speech | prototype + production |
| LibriSpeech | Clean speech | production only |
| MUSAN | Noise & music | prototype + production |
| FSD50K | Noise (CC0/CC-BY filtered) | prototype + production |
| AIR / OpenAIR | RIR | prototype + production |
| AcousticRooms | RIR | production only |

### Download

```bash
DATA_DIR="/path/to/data" \
PROFILE=prototype \
DOWNLOAD=1 \
AGREE_LICENSES=1 \
INSTALL_AUDB=1 \
uv run bash scripts/datasets/download_datasets.sh
```

Set `PROFILE=production` for the full training set. The `AGREE_LICENSES=1` flag is required and confirms acceptance of each dataset's original license terms.

For a full breakdown of the download pipeline, dataset structure, filtering logic, and license details, see [docs/download_datasets.md](docs/download_datasets.md).

### Exploring the Data

A Jupyter notebook is provided to inspect waveforms, spectrograms, and audio samples:

```bash
uv run jupyter notebook docs/visualize_data.ipynb
```

## Project Structure

```
LightDFN/
├── src/
│   ├── lightdeepfilternet.py  # Model definition
│   ├── modules.py             # Li-GRU and sub-modules
│   ├── erb.py                 # ERB filterbank utilities
│   ├── config.py              # ModelConfig dataclass
│   ├── augmentations.py       # Audio augmentation pipeline
│   ├── dataloader.py          # Dataset loader
│   └── utils.py               # Shared utilities
├── scripts/
│   └── datasets/              # Download & preprocessing scripts
├── docs/
│   ├── download_datasets.md   # Detailed dataset documentation
│   └── visualize_data.ipynb   # Data exploration notebook
└── pyproject.toml
```

## References

- **DeepFilterNet3** — H. Schröter et al.: [github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **Li-GRU** — M. Ravanelli et al. (SpeechBrain): [github.com/speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)
- **Dataset pipeline** — [sealad886/DeepFilterNet4](https://github.com/sealad886/DeepFilterNet4)
