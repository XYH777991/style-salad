#!/usr/bin/env bash
# Unattended evaluation of the tracked, officially released checkpoint
# (checkpoints/t2sm/ours/epoch_0100.ckpt) via the original configs/ours.yaml.
# Read-only against the checkpoint; only writes the metrics CSV. Runs on
# GPU 2 (least contended at launch time) so it doesn't queue behind the
# in-progress reruns/ours_train training+eval job on GPU 0. Started with
# setsid+nohup+disown so it survives the launching terminal closing.
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/ours.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
EVAL_LOG="${LOG_DIR}/evaluate_official.log"
STATUS_FILE="${LOG_DIR}/STATUS_official"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START evaluate (official epoch_0100.ckpt, GPU2)" | tee -a "${STATUS_FILE}"
python -m style_salad.evaluation.evaluate \
  --config "${CONFIG}" \
  --csv_name metrics_ours_official.csv \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
echo "[$(date -Is)] evaluate exit_code=${eval_rc}" | tee -a "${STATUS_FILE}"
echo "[$(date -Is)] DONE" | tee -a "${STATUS_FILE}"
exit "${eval_rc}"
