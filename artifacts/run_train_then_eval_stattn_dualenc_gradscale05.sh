#!/usr/bin/env bash
# Follow-up to the dual-encoder detach_from_generation work (#17 full
# detach, #18 no detach). #18 closed the SRA_5 gap to baseline entirely but
# cost a small, borderline-significant R-Prec@3 (z=-2.01 vs the 4-seed
# baseline). Generalized detach_from_generation (bool) into
# diffusion_grad_scale (float in [0,1], straight-through gradient scaling,
# see t2sm.py _style_embeddings) and this run tests the midpoint 0.5 --
# does it recover some R-Prec@3 while keeping most of the SRA_5 gain?
# Seed42 first as a directional check (configs/reruns/
# ours_train_stattn_dualenc_gradscale05.yaml) before committing to the
# other 3 seeds.
# Non-destructive: writes to checkpoints/t2sm/reruns/ours_train_stattn_dualenc_gradscale05/ (gitignored).
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/reruns/ours_train_stattn_dualenc_gradscale05.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
TRAIN_LOG="${LOG_DIR}/train_stattn_dualenc_gradscale05.log"
EVAL_LOG="${LOG_DIR}/evaluate_stattn_dualenc_gradscale05.log"
STATUS_FILE="${LOG_DIR}/STATUS_stattn_dualenc_gradscale05"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START train (diffusion_grad_scale=0.5, seed=42, GPU1)" | tee -a "${STATUS_FILE}"
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
  --csv_name metrics_ours_train_stattn_dualenc_gradscale05.csv \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
echo "[$(date -Is)] evaluate exit_code=${eval_rc}" | tee -a "${STATUS_FILE}"

if [ "${eval_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: train+evaluate finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: evaluate failed" | tee -a "${STATUS_FILE}"
fi
exit "${eval_rc}"
