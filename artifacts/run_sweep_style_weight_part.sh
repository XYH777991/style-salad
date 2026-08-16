#!/usr/bin/env bash
# Parameterized worker for a style_weight sweep against the tracked,
# officially released checkpoint (checkpoints/t2sm/ours/epoch_0100.ckpt) via
# configs/ours.yaml. style_guidance is left at the config default (0.75) —
# only style_weight (the classifier-free guidance weight for style, distinct
# from the latent-space style_guidance term) is varied. Several of these run
# in parallel (one per idle GPU). Each worker writes its own CSV/log/status
# file to avoid concurrent-append races; combine the CSVs afterward.
#
# Usage: run_sweep_style_weight_part.sh <gpu_id> <part_name> <value> [value...]
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/ours.yaml"

GPU_ID="$1"; shift
PART_NAME="$1"; shift
VALUES=("$@")

LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
LOG="${LOG_DIR}/sweep_style_weight_${PART_NAME}.log"
STATUS_FILE="${LOG_DIR}/STATUS_sweep_style_weight_${PART_NAME}"
CSV_NAME="metrics_ours_style_weight_sweep_${PART_NAME}.csv"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START sweep part=${PART_NAME} style_weight=${VALUES[*]} (GPU${GPU_ID})" | tee -a "${STATUS_FILE}"

overall_rc=0
for sw in "${VALUES[@]}"; do
  echo "[$(date -Is)] POINT style_weight=${sw} START" | tee -a "${STATUS_FILE}"
  python -m style_salad.evaluation.evaluate \
    --config "${CONFIG}" \
    --style_weight "${sw}" \
    --csv_name "${CSV_NAME}" \
    >> "${LOG}" 2>&1
  rc=$?
  echo "[$(date -Is)] POINT style_weight=${sw} exit_code=${rc}" | tee -a "${STATUS_FILE}"
  if [ "${rc}" -ne 0 ]; then
    overall_rc=${rc}
    echo "[$(date -Is)] ABORT: point style_weight=${sw} failed, stopping part=${PART_NAME}" | tee -a "${STATUS_FILE}"
    break
  fi
done

if [ "${overall_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: part=${PART_NAME} finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: part=${PART_NAME} failed" | tee -a "${STATUS_FILE}"
fi
exit "${overall_rc}"
