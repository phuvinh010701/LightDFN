#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DEFAULT_DATA_DIR="${ROOT_DIR}/datasets"

usage_helptext() {
  cat <<EOF
Usage:
  ./build_hdf5.sh [options]

Build the speech, noise, and RIR HDF5 datasets used for training from the
generated manifest lists.

Paths:
  --data-dir PATH                 Base dataset directory (default: ${DEFAULT_DATA_DIR})
  --list-dir PATH                 Directory containing manifest lists (default: DATA_DIR/lists)
  --hdf5-dir PATH                 Output directory for HDF5 files (default: DATA_DIR/hdf5)
  --cfg-out PATH                  Destination dataset.cfg path (default: DATA_DIR/dataset.cfg)
  --clean-list PATH               Clean speech manifest (default: LIST_DIR/clean_all.txt)
  --noise-list PATH               Noise/music manifest (default: LIST_DIR/noise_music.txt)
  --rir-list PATH                 RIR manifest (default: LIST_DIR/rir_all.txt)

Build configuration:
  --profile NAME                  prototype | production | apple (default: prototype)
  --sr N                          Target sample rate (default: 48000)
  --dtype NAME                    Output dtype for packed audio (default: int16)
  --num-workers N                 Override prepare_data worker count (default: profile-specific)
  --max-freq N                    Override max frequency (default: profile-specific)
  --subsample N                   Keep every Nth item from each manifest (default: 1)

Config template handling:
  --force-copy-cfg                Always overwrite CFG_OUT with profile template
  --no-force-copy-cfg             Only copy template when CFG_OUT is missing (default)

General:
  -h, --help                      Show this help message and exit

Environment variables remain supported as fallbacks when the equivalent CLI
flag is not provided; CLI options take precedence.

Examples:
  ./build_hdf5.sh --profile apple --subsample 10
  ./build_hdf5.sh --data-dir ./data --num-workers 8 --max-freq -1
EOF
}

fail() {
  echo "[error] $*" >&2
  exit 1
}

info() {
  echo "[info] $*"
}

require_file() {
  local label="$1"
  local path="$2"
  [[ -f "${path}" ]] || fail "${label} not found: ${path}"
}

require_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^-?[0-9]+$ ]] || fail "${name} must be an integer: ${value}"
}

sample_list() {
  local src="$1"
  local dst="$2"
  local stride="$3"
  awk "NR % ${stride} == 1" "${src}" > "${dst}"
}

copy_dataset_cfg() {
  local template_path="$1"
  local output_path="$2"
  local force_copy="$3"

  require_file "Dataset config template" "${template_path}"
  mkdir -p "$(dirname "${output_path}")"

  if [[ "${force_copy}" == "1" || ! -f "${output_path}" ]]; then
    cp "${template_path}" "${output_path}"
    info "wrote dataset config: ${output_path}"
  else
    info "keeping existing dataset config: ${output_path}"
  fi
}

build_dataset() {
  local step_label="$1"
  local dataset_kind="$2"
  local manifest_path="$3"
  local output_path="$4"
  shift 4

  echo "${step_label} Building ${dataset_kind} dataset..."
  uv run python -m scripts.datasets.prepare_data "${dataset_kind}" \
    "${manifest_path}" "${output_path}" \
    "$@"
}

CLI_DATA_DIR=""
CLI_LIST_DIR=""
CLI_HDF5_DIR=""
CLI_CFG_OUT=""
CLI_PROFILE=""
CLI_SR=""
CLI_DTYPE=""
CLI_NUM_WORKERS=""
CLI_MAX_FREQ=""
CLI_SUBSAMPLE=""
CLI_FORCE_COPY_CFG=""
CLI_CLEAN_LIST=""
CLI_NOISE_LIST=""
CLI_RIR_LIST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      CLI_DATA_DIR="$2"
      shift 2
      ;;
    --list-dir)
      CLI_LIST_DIR="$2"
      shift 2
      ;;
    --hdf5-dir)
      CLI_HDF5_DIR="$2"
      shift 2
      ;;
    --cfg-out)
      CLI_CFG_OUT="$2"
      shift 2
      ;;
    --profile)
      CLI_PROFILE="$2"
      shift 2
      ;;
    --sr)
      CLI_SR="$2"
      shift 2
      ;;
    --dtype)
      CLI_DTYPE="$2"
      shift 2
      ;;
    --num-workers)
      CLI_NUM_WORKERS="$2"
      shift 2
      ;;
    --max-freq)
      CLI_MAX_FREQ="$2"
      shift 2
      ;;
    --subsample)
      CLI_SUBSAMPLE="$2"
      shift 2
      ;;
    --force-copy-cfg)
      CLI_FORCE_COPY_CFG="1"
      shift
      ;;
    --no-force-copy-cfg)
      CLI_FORCE_COPY_CFG="0"
      shift
      ;;
    --clean-list)
      CLI_CLEAN_LIST="$2"
      shift 2
      ;;
    --noise-list)
      CLI_NOISE_LIST="$2"
      shift 2
      ;;
    --rir-list)
      CLI_RIR_LIST="$2"
      shift 2
      ;;
    -h|--help)
      usage_helptext
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

DATA_DIR="${CLI_DATA_DIR:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"
LIST_DIR="${CLI_LIST_DIR:-${LIST_DIR:-${DATA_DIR}/lists}}"
HDF5_DIR="${CLI_HDF5_DIR:-${HDF5_DIR:-${DATA_DIR}/hdf5}}"
PROFILE="${CLI_PROFILE:-${PROFILE:-prototype}}"
CFG_OUT="${CLI_CFG_OUT:-${CFG_OUT:-${DATA_DIR}/configs/${PROFILE}/dataset.cfg}}"
SR="${CLI_SR:-${SR:-48000}}"
DTYPE="${CLI_DTYPE:-${DTYPE:-int16}}"
SUBSAMPLE="${CLI_SUBSAMPLE:-${SUBSAMPLE:-1}}"
FORCE_COPY_CFG="${CLI_FORCE_COPY_CFG:-${FORCE_COPY_CFG:-0}}"

CLEAN_LIST="${CLI_CLEAN_LIST:-${CLEAN_LIST:-${LIST_DIR}/clean_all.txt}}"
NOISE_LIST="${CLI_NOISE_LIST:-${NOISE_LIST:-${LIST_DIR}/noise_music.txt}}"
RIR_LIST="${CLI_RIR_LIST:-${RIR_LIST:-${LIST_DIR}/rir_all.txt}}"

require_integer "SR" "${SR}"
require_integer "SUBSAMPLE" "${SUBSAMPLE}"
[[ "${SUBSAMPLE}" -ge 1 ]] || fail "SUBSAMPLE must be >= 1"

case "${PROFILE}" in
  prototype|production|apple)
    ;;
  *)
    fail "PROFILE must be one of: prototype, production, apple"
    ;;
esac

if [[ -n "${CLI_NUM_WORKERS:-${NUM_WORKERS:-}}" ]]; then
  NUM_WORKERS="${CLI_NUM_WORKERS:-${NUM_WORKERS:-}}"
else
  if [[ "${PROFILE}" == "apple" ]]; then
    NUM_WORKERS=2
  else
    NUM_WORKERS=4
  fi
fi

if [[ -n "${CLI_MAX_FREQ:-${MAX_FREQ:-}}" ]]; then
  MAX_FREQ="${CLI_MAX_FREQ:-${MAX_FREQ:-}}"
else
  if [[ "${PROFILE}" == "apple" ]]; then
    MAX_FREQ=24000
  else
    MAX_FREQ=-1
  fi
fi

require_integer "NUM_WORKERS" "${NUM_WORKERS}"
require_integer "MAX_FREQ" "${MAX_FREQ}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_data.py"
CFG_TEMPLATE="${ROOT_DIR}/datasets/configs/${PROFILE}/dataset.cfg"

require_file "prepare_data.py" "${PREPARE_SCRIPT}"
require_file "Clean list" "${CLEAN_LIST}"
require_file "Noise list" "${NOISE_LIST}"
require_file "RIR list" "${RIR_LIST}"

mkdir -p "${HDF5_DIR}"

SUFFIX=""
ACTIVE_CLEAN_LIST="${CLEAN_LIST}"
ACTIVE_NOISE_LIST="${NOISE_LIST}"
ACTIVE_RIR_LIST="${RIR_LIST}"

if [[ "${SUBSAMPLE}" -gt 1 ]]; then
  SUFFIX="_minisize"
  TMP_LIST_DIR="${LIST_DIR}/subsampled_${SUBSAMPLE}"
  mkdir -p "${TMP_LIST_DIR}"

  sample_list "${CLEAN_LIST}" "${TMP_LIST_DIR}/clean_sampled.txt" "${SUBSAMPLE}"
  sample_list "${NOISE_LIST}" "${TMP_LIST_DIR}/noise_sampled.txt" "${SUBSAMPLE}"
  sample_list "${RIR_LIST}" "${TMP_LIST_DIR}/rir_sampled.txt" "${SUBSAMPLE}"

  ACTIVE_CLEAN_LIST="${TMP_LIST_DIR}/clean_sampled.txt"
  ACTIVE_NOISE_LIST="${TMP_LIST_DIR}/noise_sampled.txt"
  ACTIVE_RIR_LIST="${TMP_LIST_DIR}/rir_sampled.txt"
fi

copy_dataset_cfg "${CFG_TEMPLATE}" "${CFG_OUT}" "${FORCE_COPY_CFG}"

COMMON_ARGS=(
  --sr "${SR}"
  --dtype "${DTYPE}"
  --num_workers "${NUM_WORKERS}"
  --max_freq "${MAX_FREQ}"
)

echo "=============================================="
echo "DeepFilterNet HDF5 Dataset Builder"
echo "=============================================="
echo "Profile:      ${PROFILE}"
echo "Data dir:     ${DATA_DIR}"
echo "List dir:     ${LIST_DIR}"
echo "HDF5 dir:     ${HDF5_DIR}"
echo "Config out:   ${CFG_OUT}"
echo "Sample rate:  ${SR}"
echo "Dtype:        ${DTYPE}"
echo "Workers:      ${NUM_WORKERS}"
echo "Max freq:     ${MAX_FREQ}"
if [[ "${SUBSAMPLE}" -gt 1 ]]; then
  echo "Mode:         MINISIZE (1/${SUBSAMPLE})"
else
  echo "Mode:         FULL DATASET"
fi
echo "=============================================="

build_dataset "[1/3]" "speech" "${ACTIVE_CLEAN_LIST}" "${HDF5_DIR}/speech_clean${SUFFIX}.hdf5" "${COMMON_ARGS[@]}"
build_dataset "[2/3]" "noise" "${ACTIVE_NOISE_LIST}" "${HDF5_DIR}/noise_music${SUFFIX}.hdf5" "${COMMON_ARGS[@]}"
build_dataset "[3/3]" "rir" "${ACTIVE_RIR_LIST}" "${HDF5_DIR}/rir${SUFFIX}.hdf5" "${COMMON_ARGS[@]}"

echo ""
echo "=============================================="
echo "Build complete"
echo "=============================================="
echo "HDF5 output folder: ${HDF5_DIR}"
echo "Files created:"
echo "  - speech_clean${SUFFIX}.hdf5"
echo "  - noise_music${SUFFIX}.hdf5"
echo "  - rir${SUFFIX}.hdf5"
echo "=============================================="
