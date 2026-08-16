#!/usr/bin/env bash
# Dual-encoder test: full StyleSTAttn (temporal+skeletal mixing) drives
# generation as before, but a SEPARATE, independent StyleMLP branch
# (pure_style_encoder, no mixing, own parameters) trains supcon directly and
# is combined (detached) with the mixing branch to actually condition
# HyperLoRA -- unlike style_readout (#12), which only affected the loss
# side and never touched generation.
# Motivated by artifacts/STYLE_ENCODER_RESEARCH_LOG.md #16: content-leakage
# into the mixing branch is present even at random init and barely changes
# with training, so no loss-side trick (#7-#15, all tried) can fix it --
# the only way to get a clean style signal is to not rely on the mixing
# branch's output for style at all.
# Non-destructive: writes to checkpoints/t2sm/reruns/ours_train_stattn_dualenc/.
# Started with setsid+nohup+disown so it survives the launching terminal closing.
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/reruns/ours_train_stattn_dualenc.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
TRAIN_LOG="${LOG_DIR}/train_stattn_dualenc.log"
EVAL_LOG="${LOG_DIR}/evaluate_stattn_dualenc.log"
STATUS_FILE="${LOG_DIR}/STATUS_stattn_dualenc"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START train (StyleSTAttn + dual pure_style_encoder, seed=42, GPU0)" | tee -a "${STATUS_FILE}"
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
  --csv_name metrics_ours_train_stattn_dualenc.csv \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
echo "[$(date -Is)] evaluate exit_code=${eval_rc}" | tee -a "${STATUS_FILE}"

if [ "${eval_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: train+evaluate finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: evaluate failed" | tee -a "${STATUS_FILE}"
fi
exit "${eval_rc}"
