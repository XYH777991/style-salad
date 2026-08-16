#!/usr/bin/env bash
# Unattended driver: train the EMA ablation config (loss identical to
# reruns/ours_train.yaml, seed=42, but with config.ema.enabled=true) into its
# own rerun copy, then on success automatically evaluate BOTH checkpoints it
# produces:
#   - epoch_0100_ema.ckpt  (EMA-smoothed weights)
#   - epoch_0100.ckpt      (raw weights from the same run, for a same-run
#                            raw-vs-EMA comparison, not just vs the older
#                            reruns/ours_train run)
# Designed to be started with setsid+nohup+disown so it survives the
# launching terminal (and Claude Code) closing.
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/reruns/ours_train_ema.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
TRAIN_LOG="${LOG_DIR}/train_ema.log"
EVAL_EMA_LOG="${LOG_DIR}/evaluate_ema.log"
EVAL_RAW_LOG="${LOG_DIR}/evaluate_ema_raw.log"
STATUS_FILE="${LOG_DIR}/STATUS_ema"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START train (ema, seed=42, GPU2 — shared with other jobs)" | tee -a "${STATUS_FILE}"
python -m style_salad.training.train --config "${CONFIG}" > "${TRAIN_LOG}" 2>&1
train_rc=$?
echo "[$(date -Is)] train exit_code=${train_rc}" | tee -a "${STATUS_FILE}"

if [ "${train_rc}" -ne 0 ]; then
  echo "[$(date -Is)] ABORT: training failed, skipping evaluation" | tee -a "${STATUS_FILE}"
  exit "${train_rc}"
fi

echo "[$(date -Is)] START evaluate (ema weights)" | tee -a "${STATUS_FILE}"
python -m style_salad.evaluation.evaluate \
  --config "${CONFIG}" \
  --csv_name metrics_ours_train_ema.csv \
  > "${EVAL_EMA_LOG}" 2>&1
eval_ema_rc=$?
echo "[$(date -Is)] evaluate(ema) exit_code=${eval_ema_rc}" | tee -a "${STATUS_FILE}"

echo "[$(date -Is)] START evaluate (raw weights, same run)" | tee -a "${STATUS_FILE}"
python -m style_salad.evaluation.evaluate \
  --config "${CONFIG}" \
  --checkpoint epoch_0100.ckpt \
  --csv_name metrics_ours_train_ema_raw.csv \
  > "${EVAL_RAW_LOG}" 2>&1
eval_raw_rc=$?
echo "[$(date -Is)] evaluate(raw) exit_code=${eval_raw_rc}" | tee -a "${STATUS_FILE}"

if [ "${eval_ema_rc}" -eq 0 ] && [ "${eval_raw_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: train+evaluate(ema+raw) finished successfully" | tee -a "${STATUS_FILE}"
  exit 0
else
  echo "[$(date -Is)] DONE: one or more evaluations failed" | tee -a "${STATUS_FILE}"
  exit 1
fi
