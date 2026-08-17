#!/usr/bin/env bash
# Fine-tunes the TemporalGate (models/transformer.py DenseFiLM) onto the
# existing ours_train_stattn_dualenc_nodetach checkpoint (via
# init_checkpoint_path), trained with the new loss_tempo (losses.py) --
# see memory: style-salad-tempo-not-transferred for the full design
# rationale. TemporalGate is zero-init (identity gate at step 0), so this
# is a short fine-tune (20 epochs), not a from-scratch 100-epoch run.
# Non-destructive: writes to checkpoints/t2sm/reruns/ours_train_stattn_dualenc_temporalgate_frozen/ (gitignored).
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/reruns/ours_train_stattn_dualenc_temporalgate_frozen.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
TRAIN_LOG="${LOG_DIR}/train_stattn_dualenc_temporalgate_frozen.log"
EVAL_LOG="${LOG_DIR}/evaluate_stattn_dualenc_temporalgate_frozen.log"
STATUS_FILE="${LOG_DIR}/STATUS_stattn_dualenc_temporalgate_frozen"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START fine-tune (TemporalGate ONLY (frozen backbone) + loss_tempo, 20 epochs, seed=42, GPU1)" | tee -a "${STATUS_FILE}"
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
  --csv_name metrics_ours_train_stattn_dualenc_temporalgate_frozen.csv \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
echo "[$(date -Is)] evaluate exit_code=${eval_rc}" | tee -a "${STATUS_FILE}"

if [ "${eval_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: train+evaluate finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: evaluate failed" | tee -a "${STATUS_FILE}"
fi
exit "${eval_rc}"
