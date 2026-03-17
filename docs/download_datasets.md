# Download Datasets

This document outlines the dataset preparation pipeline for LightDFN. It details how datasets are downloaded, structured, filtered, and compiled into lists ready for training.

## 1. Step-by-Step Guide

Follow these steps to download and prepare the datasets for training:

### Step 1: Install Prerequisites
Ensure you have the necessary system dependencies installed before running the script:
- Basic utilities: `curl`, `wget`, `tar`, `unzip`, `zip`, `rsync`
- **Recommended**: Download accelerator: `aria2`
- (Optional): GitHub CLI (`gh`) for faster and authenticated downloads from GitHub.

On Ubuntu/Debian, you can install the essentials via:
```bash
sudo apt-get update
sudo apt-get install curl wget tar unzip zip rsync aria2
```

### Step 2: Execute the Download Script
Configure the download behavior by setting environment variables before running the script. You must explicitly agree to the dataset licenses to proceed.

```bash
DATA_DIR="./datasets" \
PROFILE="prototype" \
DOWNLOAD=1 \
AGREE_LICENSES=1 \
INSTALL_AUDB=1 \
USE_ARIA2=1 \
uv run bash scripts/datasets/download_datasets.sh
```

**Environment Variables Explained:**
- `DATA_DIR`: The target directory where all files (archives, raw audio, lists) will be stored.
- `PROFILE`: Determines the size/scope of the dataset to be downloaded. 
  - `"prototype"`: (Default) Downloads a smaller subset of data (VCTK, FSD50K, MUSAN, AIR, OpenAIR). Best for local testing and quick iterations.
  - `"production"`: Downloads the full comprehensive dataset mix including full LibriSpeech and AcousticRooms. **Warning**: This requires hundreds of GBs of storage.
- `DOWNLOAD`: Set to `1` to enable downloading dataset archives. If set to `0`, it will skip downloading and only attempt to recreate the file lists based on existing data.
- `AGREE_LICENSES`: Must be set to `1` to confirm you have read and agreed to the original dataset licenses.
- `INSTALL_AUDB`: Set to `1` to automatically install the `audb` Python library to fetch the AIR and OpenAIR datasets.
- `USE_ARIA2`: Set to `1` to speed up large downloads using `aria2c` as an accelerator (if installed).

### Step 3: Verify the Output
Once the script completes, it will generate text files in the `datasets/lists/` directory containing the absolute paths to all valid audio files. Specifically, you should check for:
- `clean_all.txt` (Combined Clean Speech)
- `noise_music.txt` (Combined Noise & Music)
- `rir_all.txt` (Combined Room Impulse Responses)

If these files are generated and not empty, the dataset preparation was successful and you are ready to build the HDF5 dataset.

## 2. Flow Download Datasets

The `scripts/datasets/download_datasets.sh` script automates the retrieval and extraction of 6 distinct speech, noise, and Room Impulse Response (RIR) datasets. The execution flow for each dataset is as follows:

### VCTK (Clean Speech)
- **Download**: Fetches `VCTK-Corpus-0.92.zip` from `datashare.is.ed.ac.uk`.
- **Extraction**: Since the archive lacks a root directory, the script creates `raw/VCTK-Corpus-0.92/` and extracts the contents directly into it.
- **Processing (File List)**: Scans for `*.flac` files (typically within `wav48_silence_trimmed`) and saves absolute paths to `lists/vctk_clean.txt`.

### LibriSpeech (Clean Speech)
- **Download**: Retrieves archives from `openslr.org/resources/12/`.
  - `PROFILE=prototype`: **Skipped entirely** (does not download).
  - `PROFILE=production`: Downloads the full dataset (`train-clean-100`, `train-clean-360`, `train-other-500`, `dev-clean`, `dev-other`, `test-clean`, `test-other`).
- **Extraction**: Extracts `.tar.gz` files directly into `raw/LibriSpeech/`.
- **Processing (File List)**: Scans for `*.flac` files and saves paths to `lists/librispeech_clean.txt`.

### MUSAN (Noise & Music)
- **Download**: Fetches `musan.tar.gz` from `openslr.org/resources/17/`.
- **Extraction**: Extracts into `raw/musan/`.
- **Processing (File List)**: Separated into two distinct lists:
  - Scans the `noise/` directory for `*.wav` -> `lists/musan_noise.txt`.
  - Scans the `music/` directory for `*.wav` -> `lists/musan_music.txt`.

### FSD50K (General Sound Events - Noise)
- **Download**: Sourced from `zenodo.org`. This dataset requires complex handling:
  - Downloads metadata archives (`metadata.zip`, `ground_truth.zip`, `doc.zip`).
  - Downloads split audio archives (`dev_audio` contains `.z01` to `.z05` + `.zip`; `eval_audio` contains `.z01` + `.zip`).
- **Merge & Extraction**: The script calls the Python helper `scripts/datasets/zip_merge_progress.py` to merge split parts into `*.merged.zip` before extracting into `raw/FSD50K/`.
- **Processing (Filtering)**: Valid paths are saved to `lists/fsd50k_filtered.txt` using a specialized Python filter (details in Section 4).

### AIR & OpenAIR (Room Impulse Responses - Reverb)
- **Download**: Managed via the `audb` Python package rather than direct URLs. The shell script invokes `scripts/datasets/audb_download.py` to fetch the databases.
- **Extraction**: Extracted by `audb` into `raw/audb/` (AIR in `data/`, OpenAIR in `wav/`).
- **Processing (File List)**:
  - Scans AIR for `*.wav` -> `lists/air_rir.txt`.
  - Scans OpenAIR for `*.wav` -> `lists/openair_rir.txt`.

### AcousticRooms (Room Impulse Responses - Reverb)
- **Download**: Fetches `single_channel_ir.zip` and `metadata.zip` from the [facebookresearch/AcousticRooms](https://github.com/facebookresearch/AcousticRooms) repository.
  - `PROFILE=prototype`: **Skipped entirely** (does not download).
  - `PROFILE=production`: Downloads the dataset.
- **Nested Extraction**: Extracted into `raw/AcousticRooms/`. The script detects that `single_channel_ir_1/` contains numerous smaller nested `.zip` files. It performs a secondary extraction to unpack these nested files in place.
- **Processing (File List)**: Scans for all `*.wav` files -> `lists/acousticrooms_rir.txt`.

## 3. Dataset Structure

After downloading and extracting, your base output directory (`/datasets`) will exhibit the following structure:

```text
/datasets/
├── downloads/               # Compressed archives & integrity caches
│   ├── .verify_cache.tsv    # Checksum cache to skip redundant verification
│   ├── VCTK-Corpus-0.92.zip
│   ├── musan.tar.gz
│   └── ...
├── raw/                     # Extracted, raw dataset files
│   ├── VCTK-Corpus-0.92/    # VCTK clean speech
│   ├── LibriSpeech/         # LibriSpeech clean speech
│   ├── musan/               # MUSAN noise and music
│   ├── FSD50K/              # Free Sound Dataset 50K
│   ├── audb/                # AIR and OpenAIR RIRs
│   └── AcousticRooms/       # AcousticRooms RIRs
└── lists/                   # Output file manifests
    ├── vctk_clean.txt
    ├── librispeech_clean.txt
    ├── musan_noise.txt
    ├── musan_music.txt
    ├── fsd50k_filtered.txt
    ├── air_rir.txt
    ├── openair_rir.txt
    ├── acousticrooms_rir.txt
    ├── clean_all.txt        # Combined Clean Speech (VCTK + LibriSpeech)
    ├── noise_music.txt      # Combined Noise & Music (MUSAN + Filtered FSD50K)
    └── rir_all.txt          # Combined Room Impulse Responses (AIR + OpenAIR + AcousticRooms)
```

## 4. License

LightDFN does not redistribute audio directly. You must accept and comply with the original licenses of the respective datasets:

- **VCTK**: [Creative Commons Attribution 4.0 International](https://datashare.ed.ac.uk/handle/10283/3443)
- **LibriSpeech**: [CC BY 4.0](https://www.openslr.org/12/)
- **MUSAN**: [CC BY 4.0](https://www.openslr.org/17/)
- **FSD50K**: Varied mixed licenses. The download pipeline explicitly enforces filtering to retain only commercially permissive subsets.
- **AIR**/**OpenAIR**: MIT License (see `raw/audb/db.yaml`)
- **AcousticRooms**: [License from facebookresearch/AcousticRooms](https://github.com/facebookresearch/AcousticRooms/blob/clean-main/LICENSE)

*Note: The script requires the flag `AGREE_LICENSES=1` to proceed, enforcing user acknowledgement of these terms.*

## 5. Filter / Post Process

Post-processing is crucial to ensure datasets meet quality and licensing requirements.

**FSD50K License Filtering (`fsd50k_filter.py`)**
Because FSD50K aggregates clips from Freesound with mixed licenses, the pipeline cannot blindly use all audio files. 
1. The script accesses `dev_clips_info_FSD50K.json`/`.csv` and `eval_clips_info_FSD50K.json`/`.csv` in the extracted metadata directory.
2. It evaluates the `license` column for every audio clip.
3. Only clips under strictly permissive licenses (**CC0** and **CC-BY**) are appended to the final `fsd50k_filtered.txt` manifest.

## 6. Explore the Data

To gain a better understanding of the datasets, check out the provided Jupyter Notebook: [docs/visualize_data.ipynb](visualize_data.ipynb). 

This notebook allows you to:
- Visually inspect the structure and metadata of the filtered data.
- Plot waveforms and spectrograms.
- Listen to various sample clips (Clean Speech, Noise, and Room Impulse Responses).