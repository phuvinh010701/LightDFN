# LightDeepFilterNet (LightDFN)

Efficient implementation of audio noise suppression using Li-GRU, based on DeepFilterNet3.

## Installation

```bash
# Clone the repository (dataset branch)
git clone https://github.com/phuvinh010701/LightDFN.git -b feat/dataset
cd LightDFN

# Install dependencies using uv
uv sync
```

## Dataset Setup

The project provides an automated script to download and prepare the speech, noise, and RIR (Room Impulse Response) corpora required for training.

### Download Command

Run the following command to start the automated download and extraction process:

```bash
DATA_DIR="/path/to/data" \
PROFILE=prototype \
DOWNLOAD=1 \
AGREE_LICENSES=1 \
INSTALL_AUDB=1 \
uv run bash scripts/datasets/download_datasets.sh
```

### Profile Comparison

The `PROFILE` variable determines which datasets are included. Use `prototype` for quick testing and `production` for full model training.

| Dataset | Type | Prototype | Production |
| :--- | :--- | :---: | :---: |
| **VCTK** | Speech (Clean) | ✅ | ✅ |
| **MUSAN** | Noise & Music | ✅ | ✅ |
| **FSD50K** | General Sound Events | ✅ | ✅ |
| **AIR / OpenAIR** | RIR (Reverb) | ✅ | ✅ |
| **LibriSpeech** | Speech (Clean) | ❌ | ✅ |
| **AcousticRooms** | RIR (Reverb) | ❌ | ✅ |

### Notes:

- **`AGREE_LICENSES=1`**: You must explicitly set this to confirm acceptance of the dataset licenses.
- **`PROFILE=production`**: Recommended for training high-quality models; includes significantly more data (LibriSpeech/AcousticRooms).
- **`INSTALL_AUDB=1`**: Automatically installs `audb`, a tool required to download the AIR and OpenAIR RIR datasets.
- **Dependencies**: Ensure `aria2`, `zip`, and `unzip` are installed. The script uses `aria2` for high-speed, parallel downloads by default.
- **Resuming**: The script caches verification results in `downloads/.verify_cache.tsv` to avoid re-scanning or re-downloading existing archives on subsequent runs.

## References

- Original DeepFilterNet3: https://github.com/Rikorose/DeepFilterNet
- Li-GRU (SpeechBrain): https://github.com/speechbrain/speechbrain
- Dataset Logic & Guide: [sealad886/DeepFilterNet4](https://github.com/sealad886/DeepFilterNet4)
