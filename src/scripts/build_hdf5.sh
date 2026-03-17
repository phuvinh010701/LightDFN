#!/usr/bin/env bash
set -euo pipefail

# Build HDF5 datasets for DeepFilterNet training.
# Requires: a Python environment with DeepFilterNet deps installed.
# pip install DeepFilterNet4[train,eval,...]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
LIST_DIR="${LIST_DIR:-${DATA_DIR}/lists}"
HDF5_DIR="${HDF5_DIR:-${DATA_DIR}/hdf5}"
PROFILE="${PROFILE:-prototype}"
CFG_OUT="${CFG_OUT:-${DATA_DIR}/dataset.cfg}"
SR="${SR:-48000}"
DTYPE="${DTYPE:-int16}"
FORCE_COPY_CFG="${FORCE_COPY_CFG:-0}"
QUIET="${QUIET:-1}"  # 1 = logs to file only, 0 = logs to terminal too

SUBSAMPLE="${SUBSAMPLE:-1}" 
SUFFIX=""
if [[ "${SUBSAMPLE}" -gt 1 ]]; then
  SUFFIX="_minisize"
fi

# Log file configuration
LOG_DIR="${LOG_DIR:-${DATA_DIR}/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

CLEAN_LIST="${CLEAN_LIST:-${LIST_DIR}/clean_all.txt}"
NOISE_LIST="${NOISE_LIST:-${LIST_DIR}/noise_music.txt}"
RIR_LIST="${RIR_LIST:-${LIST_DIR}/rir_all.txt}"

CFG_TEMPLATE="${ROOT_DIR}/datasets/${PROFILE}/dataset.cfg"

mkdir -p "${HDF5_DIR}"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "DeepFilterNet HDF5 Dataset Builder"
echo "=============================================="
echo "Profile:    ${PROFILE}"
if [[ "${SUBSAMPLE}" -gt 1 ]]; then
  echo "Mode:       MINISIZE (1/${SUBSAMPLE})"
else
  echo "Mode:       FULL DATASET"
fi
echo "=============================================="

if [[ "${SUBSAMPLE}" -gt 1 ]]; then
  TMP_LIST_DIR="${LIST_DIR}/subsampled_${SUBSAMPLE}"
  mkdir -p "${TMP_LIST_DIR}"

  sample_list() {
    local src=$1
    local dst=$2
    awk "NR % ${SUBSAMPLE} == 1" "${src}" > "${dst}"
  }

  sample_list "${CLEAN_LIST}" "${TMP_LIST_DIR}/clean_sampled.txt"
  CLEAN_LIST="${TMP_LIST_DIR}/clean_sampled.txt"

  sample_list "${NOISE_LIST}" "${TMP_LIST_DIR}/noise_sampled.txt"
  NOISE_LIST="${TMP_LIST_DIR}/noise_sampled.txt"

  sample_list "${RIR_LIST}" "${TMP_LIST_DIR}/rir_sampled.txt"
  RIR_LIST="${TMP_LIST_DIR}/rir_sampled.txt"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_data.py"

if [[ ! -f "${PREPARE_SCRIPT}" ]]; then
  echo "[error] Không tìm thấy file ${PREPARE_SCRIPT}" >&2
  exit 1
fi

if [[ "${PROFILE}" == "apple" ]]; then
  NUM_WORKERS="${NUM_WORKERS:-2}"
  MAX_FREQ="${MAX_FREQ:-24000}"
else
  NUM_WORKERS="${NUM_WORKERS:-4}"
  MAX_FREQ="${MAX_FREQ:--1}"
fi

COMMON_ARGS="--sr ${SR} --dtype ${DTYPE} --num_workers ${NUM_WORKERS} --max_freq ${MAX_FREQ}"
if [[ "${QUIET}" == "1" ]]; then
  COMMON_ARGS="${COMMON_ARGS} --quiet"
fi

echo ""
echo "[1/3] Building speech dataset..."
python "${PREPARE_SCRIPT}" speech \
  "${CLEAN_LIST}" "${HDF5_DIR}/speech_clean${SUFFIX}.hdf5" \
  ${COMMON_ARGS}

echo ""
echo "[2/3] Building noise dataset..."
python "${PREPARE_SCRIPT}" noise \
  "${NOISE_LIST}" "${HDF5_DIR}/noise_music${SUFFIX}.hdf5" \
  ${COMMON_ARGS}

echo ""
echo "[3/3] Building RIR dataset..."
python "${PREPARE_SCRIPT}" rir \
  "${RIR_LIST}" "${HDF5_DIR}/rir${SUFFIX}.hdf5" \
  ${COMMON_ARGS}

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo "HDF5 output folder: ${HDF5_DIR}"
echo "Files created:"
echo "  - speech_clean${SUFFIX}.hdf5"
echo "  - noise_music${SUFFIX}.hdf5"
echo "  - rir${SUFFIX}.hdf5"
echo "=============================================="