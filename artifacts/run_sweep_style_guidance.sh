#!/usr/bin/env bash
# Sweep config.model.style_guidance against the tracked, officially released
# checkpoint (checkpoints/t2sm/ours/epoch_0100.ckpt) via configs/ours.yaml,
# to see whether a different guidance weight reproduces the paper's reported
# SRA (~76.03) without needing to retrain anything. Read-only against the
# checkpoint; each point appends one row to the sweep CSV via
# evaluate.py's --style_guidance override. style_weight is left at the
# config default (1.5) — only style_guidance is varied.
#
# Runs on GPU2 (idle at launch time). Started with setsid+nohup+disown so it
# survives the launching terminal (and Claude Code) closing.
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/ours.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
LOG="${LOG_DIR}/sweep_style_guidance.log"
STATUS_FILE="${LOG_DIR}/STATUS_sweep_style_guidance"
CSV_NAME="metrics_ours_style_guidance_sweep.csv"

# 0.75 is already covered by artifacts/evaluation/metrics_ours_official.csv
# (SRA=57.37); not repeated here.
VALUES=(0.1 0.25 0.5 1.0 1.5 2.0 3.0)

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START sweep style_guidance=${VALUES[*]} (GPU2)" | tee -a "${STATUS_FILE}"

overall_rc=0
for sg in "${VALUES[@]}"; do
  echo "[$(date -Is)] POINT style_guidance=${sg} START" | tee -a "${STATUS_FILE}"
  python -m style_salad.evaluation.evaluate \
    --config "${CONFIG}" \
    --style_guidance "${sg}" \
    --csv_name "${CSV_NAME}" \
    >> "${LOG}" 2>&1
  rc=$?
  echo "[$(date -Is)] POINT style_guidance=${sg} exit_code=${rc}" | tee -a "${STATUS_FILE}"
  if [ "${rc}" -ne 0 ]; then
    overall_rc=${rc}
    echo "[$(date -Is)] ABORT: point style_guidance=${sg} failed, stopping sweep" | tee -a "${STATUS_FILE}"
    break
  fi
done

if [ "${overall_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: sweep finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: sweep failed" | tee -a "${STATUS_FILE}"
fi
exit "${overall_rc}"
