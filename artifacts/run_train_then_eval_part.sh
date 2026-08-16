#!/usr/bin/env bash
# Parameterized train-then-evaluate worker (hard SupCon baseline seed sweep).
# Used to add more seeds to the baseline distribution alongside the existing
# reruns/ours_train (seed42) and reruns/ours_train_seed123 runs, so future
# ablations have a proper baseline mean/variance to compare against instead
# of a single point estimate. Started with setsid+nohup+disown so it
# survives the launching terminal (and Claude Code) closing.
#
# Usage: run_train_then_eval_part.sh <gpu_id> <config_relpath> <csv_name>
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"

GPU_ID="$1"
CONFIG="$2"
CSV_NAME="$3"
TAG="$(basename "${CONFIG}" .yaml)"

LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
TRAIN_LOG="${LOG_DIR}/train_${TAG}.log"
EVAL_LOG="${LOG_DIR}/evaluate_${TAG}.log"
STATUS_FILE="${LOG_DIR}/STATUS_${TAG}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START train (${TAG}, GPU${GPU_ID})" | tee -a "${STATUS_FILE}"
python -m style_salad.training.train --config "${CONFIG}" > "${TRAIN_LOG}" 2>&1
train_rc=$?
echo "[$(date -Is)] train exit_code=${train_rc}" | tee -a "${STATUS_FILE}"

if [ "${train_rc}" -ne 0 ]; then
  echo "[$(date -Is)] ABORT: training failed, skipping evaluation" | tee -a "${STATUS_FILE}"
  exit "${train_rc}"
fi

echo "[$(date -Is)] START evaluate" | tee -a "${STATUS_FILE}"
python -m style_salad.evaluation.evaluate \
  --config "${CONFIG}" \
  --csv_name "${CSV_NAME}" \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
echo "[$(date -Is)] evaluate exit_code=${eval_rc}" | tee -a "${STATUS_FILE}"

if [ "${eval_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: train+evaluate finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: evaluate failed" | tee -a "${STATUS_FILE}"
fi
exit "${eval_rc}"
