#!/usr/bin/env bash
# Unattended driver: train the soft-SupCon ablation config (loss.soft_supcon
# instead of loss.supcon) into its own rerun copy, then on success
# automatically launch evaluation against the freshly trained checkpoint.
# Designed to be started with setsid+nohup+disown so it survives the
# launching terminal (and Claude Code) closing.
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/reruns/ours_train_soft_supcon.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
TRAIN_LOG="${LOG_DIR}/train_soft_supcon.log"
EVAL_LOG="${LOG_DIR}/evaluate_soft_supcon.log"
STATUS_FILE="${LOG_DIR}/STATUS_soft_supcon"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

# Avoid the flaky local proxy; nothing here needs it (assets are already local).
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# CLIP weights are pre-cached under checkpoints/clip-vit-base-patch32/; force
# offline mode so it never blocks retrying the (proxy-blocked) huggingface.co.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# GPU1 was idle (0 MiB used) when this run was launched; other GPUs on this
# host had other users'/runs' processes on them.
export CUDA_VISIBLE_DEVICES=1

echo "[$(date -Is)] START train (loss=soft_supcon, GPU1)" | tee -a "${STATUS_FILE}"
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
  --csv_name metrics_ours_soft_supcon.csv \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
echo "[$(date -Is)] evaluate exit_code=${eval_rc}" | tee -a "${STATUS_FILE}"

if [ "${eval_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: train+evaluate finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: evaluate failed" | tee -a "${STATUS_FILE}"
fi
exit "${eval_rc}"
