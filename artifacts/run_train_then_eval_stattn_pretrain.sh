#!/usr/bin/env bash
# Staged training test: full StyleSTAttn (temporal+skeletal attn), seed=42,
# no content_adversary, no style_readout -- but with a new Phase 1
# (pretrain_style_epochs: 20, configs/reruns/ours_train_stattn_pretrain.yaml):
# style_encoder is pretrained with supcon ALONE (denoiser not even run) for
# 20 epochs before Phase 2 (the unchanged, original 100-epoch joint
# training) begins. Tests whether giving supcon uncontested control of the
# trunk BEFORE the diffusion objective ever competes for it rescues SRA --
# both raising supcon's weight (metrics_ours_train_stattn_supcon4.csv) and
# architecturally detaching it from the trunk
# (metrics_ours_train_stattn_splithead.csv) left SRA essentially unchanged,
# suggesting supcon's gradient into the trunk was never large enough to
# compete in the first place, however it was weighted or isolated -- staging
# sidesteps the competition question by controlling WHEN each objective
# gets to shape the trunk instead of how much or how isolated.
# Non-destructive: writes to checkpoints/t2sm/reruns/ours_train_stattn_pretrain/.
# Started with setsid+nohup+disown so it survives the launching terminal closing.
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/reruns/ours_train_stattn_pretrain.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
TRAIN_LOG="${LOG_DIR}/train_stattn_pretrain.log"
EVAL_LOG="${LOG_DIR}/evaluate_stattn_pretrain.log"
STATUS_FILE="${LOG_DIR}/STATUS_stattn_pretrain"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START train (StyleSTAttn + 20-epoch supcon-only pretrain phase, seed=42, GPU0)" | tee -a "${STATUS_FILE}"
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
  --csv_name metrics_ours_train_stattn_pretrain.csv \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
echo "[$(date -Is)] evaluate exit_code=${eval_rc}" | tee -a "${STATUS_FILE}"

if [ "${eval_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: train+evaluate finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: evaluate failed" | tee -a "${STATUS_FILE}"
fi
exit "${eval_rc}"
