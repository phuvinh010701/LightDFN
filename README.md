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

LightDFN uses six corpora spanning clean speech, noise, music, and room impulse responses (RIRs). The dataset tooling handles download, extraction, license-aware filtering, and manifest generation.

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
./script/datasets/download_datasets.sh \
  --data-dir "./datasets/" \
  --profile prototype \
  --use-aria2 \
  --aria2-max-concurrent 4 \
  --aria2-conn 8 \
  --resume \
  --install-audb
```

Use `--profile production` for the full training set. `--agree-licenses` is required and confirms acceptance of each dataset's original license terms.

For the full download workflow, dataset layout, filtering rules, and licensing notes, see `docs/download_datasets.md`.

### Preprocessing

After the raw datasets and manifests are available, build the training HDF5 files with:

```bash
scripts/datasets/build_hdf5.sh \
  --data-dir "./datasets/" \
  --num-workers 4 \
  --dtype float32 \
  --profile prototype
```

For preprocessing details, profile behavior, and output layout, see `docs/preprocess_data.md`.

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
│   ├── preprocess_data.md     # Preprocessing workflow documentation
│   └── visualize_data.ipynb   # Data exploration notebook
└── pyproject.toml
```

## References

- **DeepFilterNet3** — H. Schröter et al.: [github.com/Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **Li-GRU** — M. Ravanelli et al. (SpeechBrain): [github.com/speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)
- **Dataset pipeline** — [sealad886/DeepFilterNet4](https://github.com/sealad886/DeepFilterNet4)
