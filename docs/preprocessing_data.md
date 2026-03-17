# Preprocessing Data

This document outlines the data preprocessing pipeline for LightDFN. After downloading the datasets, the raw audio files must be packed into HDF5 format to enable efficient DataLoader streaming, fast I/O, and caching during training.

## 1. Flow Build HDF5

The `scripts/datasets/build_hdf5.sh` script automates the generation of HDF5 datasets from the previously generated file lists. It also configures system parameters and hardware behavior based on the chosen dataset profile.

### Initialization & Configuration
- **Profile Setup**: Reads the `PROFILE` environment variable (`prototype`, `production`, or `apple`).
- **Dataset Configuration**: Copies the profile-specific configuration template from `datasets/${PROFILE}/dataset.cfg` to the target data directory (`data/dataset.cfg`) if it does not already exist. This file defines the dataloader rules and mixing parameters during training.
- **Hardware Tuning**: 
  - For standard systems (`prototype`, `production`): Defaults to 4 parallel workers (`NUM_WORKERS=4`) and preserves the full frequency (`MAX_FREQ=-1`).
  - For Apple Silicon (`apple`): Reduces parallel processing to 2 workers (`NUM_WORKERS=2`) and caps the maximum frequency (`MAX_FREQ=24000`) to manage memory pressure gracefully.

### Clean Speech Dataset
- **Input Manifest**: Consumes `lists/clean_all.txt` (e.g., VCTK and LibriSpeech).
- **Processing Engine**: Invokes the `df.scripts.prepare_data speech` module, which performs several advanced operations:
  - **Path Validation**: Pre-validates all file paths in the list using a fast multiprocessing pool.
  - **Incremental Builds**: Checks the output HDF5 to skip audio files that have already been packaged.
  - **Decoding & Resampling**: Uses the high-performance `torchcodec` library (`AudioDecoder`) to decode and resample audio on-the-fly to the target rate (default 48kHz).
  - **Encoding**: Pushes parallel batches through PyTorch's `DataLoader` and encodes them to the target storage format (e.g., `pcm` `int16`, `flac`, or `vorbis`) via `AudioEncoder`.
- **Output**: Saves the packed container as `hdf5/speech_clean.hdf5` with metadata attributes (db_id, sr, max_freq, codec, etc.) and uses a directory-flattened relative path as the HDF5 internal structure key.

### Noise & Music Dataset
- **Input Manifest**: Consumes `lists/noise_music.txt` (e.g., MUSAN and FSD50K).
- **Processing**: Invokes `df.scripts.prepare_data noise` utilizing the identical `torchcodec` extraction and multiprocessing engine as the speech pipeline.
- **Output**: Saves the packed container as `hdf5/noise_music.hdf5`.

### Room Impulse Response (RIR) Dataset
- **Input Manifest**: Consumes `lists/rir_all.txt` (e.g., AIR, OpenAIR, and AcousticRooms).
- **Processing**: Invokes `df.scripts.prepare_data rir` to resample and pack the reverberation kernels into a single block.
- **Output**: Saves the packed container as `hdf5/rir.hdf5`.

---

## 2. Dataset Structure

After a successful build, your base data directory (`<DATA_DIR>`) will exhibit the following structure with the newly generated HDF5 representations:

```text
<DATA_DIR>/
├── dataset.cfg              # Dataloader training configuration for the selected profile
├── logs/                    # Directory containing build execution logs
│   └── build_hdf5_*.log     # Detailed logs for each HDF5 dataset generation run
└── hdf5/                    # Packed HDF5 datasets consumed directly by the training pipeline
    ├── speech_clean.hdf5    
    ├── noise_music.hdf5     
    └── rir.hdf5             
```

---

## 3. Customizations

Advanced users can override the default preprocessing parameters directly through environment variables when running the build script:

- `SR`: Target sample rate for the packed dataset (default: `48000`).
- `DTYPE`: Audio sample data type (default: `int16`).
- `NUM_WORKERS`: Number of CPU workers for reading and packing audio.
- `FORCE_COPY_CFG=1`: Forces the script to overwrite the existing `dataset.cfg` in the target directory with the pristine profile template.
- `QUIET=0`: Pipes detailed logs from the DeepFilterNet Python worker directly to the terminal stdout (by default, logs are only written to the log file to reduce noise).
