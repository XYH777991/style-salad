#!/usr/bin/env bash
# Architectural separation test: full StyleSTAttn (temporal+skeletal attn),
# seed=42, no content_adversary, but style_readout is now configured
# (configs/reruns/ours_train_stattn_splithead.yaml). supcon now reads a
# DETACHED + reprojected copy of the pooled embedding (t2sm.py
# _apply_style_readout) instead of the raw embedding that drives HyperLoRA --
# its gradient can only reshape style_readout's own small weights, never the
# style_encoder / mixing trunk. The trunk stays driven purely by the
# diffusion objective (preserving the FID/R-Precision gains from cross-cell
# mixing); the readout head is free to organize whatever style signal
# survives without fighting the trunk for it (AutoVC-style architectural
# bypass, see artifacts/STYLE_ENCODER_RESEARCH_LOG.md section 6 -- five
# loss-balancing attempts on a SHARED vector all failed).
# Non-destructive: writes to checkpoints/t2sm/reruns/ours_train_stattn_splithead/.
# Started with setsid+nohup+disown so it survives the launching terminal closing.
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/reruns/ours_train_stattn_splithead.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
TRAIN_LOG="${LOG_DIR}/train_stattn_splithead.log"
EVAL_LOG="${LOG_DIR}/evaluate_stattn_splithead.log"
STATUS_FILE="${LOG_DIR}/STATUS_stattn_splithead"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START train (StyleSTAttn + split style_readout head, seed=42, GPU0)" | tee -a "${STATUS_FILE}"
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
  --csv_name metrics_ours_train_stattn_splithead.csv \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
echo "[$(date -Is)] evaluate exit_code=${eval_rc}" | tee -a "${STATUS_FILE}"

if [ "${eval_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: train+evaluate finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: evaluate failed" | tee -a "${STATUS_FILE}"
fi
exit "${eval_rc}"
