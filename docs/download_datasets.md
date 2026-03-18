# Download Datasets

This document describes the dataset acquisition workflow for LightDFN. The pipeline downloads the required corpora, extracts them into a consistent directory layout, applies post-processing where necessary, and generates manifest files for preprocessing and training.

## Command

Use the repository script directly with CLI flags:

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

Use `--profile production` for the full corpus mix. If license confirmation is not already enabled in your local defaults, add `--agree-licenses`.

## Workflow

The `scripts/datasets/download_datasets.sh` script orchestrates six datasets covering clean speech, background noise, music, and room impulse responses.

### VCTK (Clean Speech)
- **Download**: Fetches `VCTK-Corpus-0.92.zip` from `datashare.is.ed.ac.uk`.
- **Extraction**: Since the archive lacks a root directory, the script creates `raw/VCTK-Corpus-0.92/` and extracts the contents directly into it.
- **Processing (File List)**: Scans for `*.flac` files (typically within `wav48_silence_trimmed`) and saves absolute paths to `lists/vctk_clean.txt`.

### LibriSpeech (Clean Speech)
- **Download**: Retrieves archives from `openslr.org/resources/12/`.
  - `--profile prototype`: skipped by default.
  - `--profile production`: downloads the full dataset (`train-clean-100`, `train-clean-360`, `train-other-500`, `dev-clean`, `dev-other`, `test-clean`, `test-other`).
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
  - `--profile prototype`: skipped by default.
  - `--profile production`: downloads the dataset.
- **Nested Extraction**: Extracted into `raw/AcousticRooms/`. The script detects that `single_channel_ir_1/` contains numerous smaller nested `.zip` files. It performs a secondary extraction to unpack these nested files in place.
- **Processing (File List)**: Scans for all `*.wav` files -> `lists/acousticrooms_rir.txt`.

## Directory Layout

After download and extraction, the base output directory typically looks like this:

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

## Licensing

LightDFN does not redistribute audio directly. You must accept and comply with the original licenses of the respective datasets:

- **VCTK**: [Creative Commons Attribution 4.0 International](https://datashare.ed.ac.uk/handle/10283/3443)
- **LibriSpeech**: [CC BY 4.0](https://www.openslr.org/12/)
- **MUSAN**: [CC BY 4.0](https://www.openslr.org/17/)
- **FSD50K**: Varied mixed licenses. The download pipeline explicitly enforces filtering to retain only commercially permissive subsets.
- **AIR**/**OpenAIR**: MIT License (see `raw/audb/db.yaml`)
- **AcousticRooms**: [License from facebookresearch/AcousticRooms](https://github.com/facebookresearch/AcousticRooms/blob/clean-main/LICENSE)

The download script requires `--agree-licenses` before it will fetch any dataset.

## Filtering And Post-Processing

Post-processing is crucial to ensure datasets meet quality and licensing requirements.

**FSD50K License Filtering (`fsd50k_filter.py`)**
Because FSD50K aggregates clips from Freesound with mixed licenses, the pipeline cannot use all audio files without filtering.
1. The script accesses `dev_clips_info_FSD50K.json`/`.csv` and `eval_clips_info_FSD50K.json`/`.csv` in the extracted metadata directory.
2. It evaluates the `license` column for every audio clip.
3. Only clips under strictly permissive licenses (**CC0** and **CC-BY**) are appended to the final `fsd50k_filtered.txt` manifest.

The helper scripts are invoked with explicit CLI flags, including `--fsd50k-dir`, `--list-dir`, `--name`, `--version`, and `--root`, so the documented workflow does not depend on environment variables.

## Explore The Data

To inspect the prepared datasets interactively, open `docs/visualize_data.ipynb`.

This notebook allows you to:
- Visually inspect the structure and metadata of the filtered data.
- Plot waveforms and spectrograms.
- Listen to various sample clips (Clean Speech, Noise, and Room Impulse Responses).
