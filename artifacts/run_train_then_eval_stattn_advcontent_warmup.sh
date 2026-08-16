#!/usr/bin/env bash
# Same as run_train_then_eval_stattn_advcontent.sh but with the GRL lambda
# warmup schedule now baked into train.py (ramps 0 -> content_adversary.lambda_
# following the standard DANN schedule 2/(1+exp(-10p))-1, instead of full
# strength from step 1). The fixed-lambda run left the content classifier
# stuck at the random-guess baseline the whole 100 epochs (raw loss
# 1.9307 -> 1.9176, vs ln(7)=1.9459) -- this tests whether giving it room to
# actually learn first lets the adversarial pressure do anything.
# Non-destructive: writes to checkpoints/t2sm/reruns/ours_train_stattn_advcontent_warmup/.
# Started with setsid+nohup+disown so it survives the launching terminal closing.
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/reruns/ours_train_stattn_advcontent_warmup.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
TRAIN_LOG="${LOG_DIR}/train_stattn_advcontent_warmup.log"
EVAL_LOG="${LOG_DIR}/evaluate_stattn_advcontent_warmup.log"
STATUS_FILE="${LOG_DIR}/STATUS_stattn_advcontent_warmup"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START train (StyleSTAttn + content-adversarial w/ GRL warmup, seed=42, GPU0)" | tee -a "${STATUS_FILE}"
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
  --csv_name metrics_ours_train_stattn_advcontent_warmup.csv \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
echo "[$(date -Is)] evaluate exit_code=${eval_rc}" | tee -a "${STATUS_FILE}"

if [ "${eval_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: train+evaluate finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: evaluate failed" | tee -a "${STATUS_FILE}"
fi
exit "${eval_rc}"
