#!/usr/bin/env bash
# Ablation on the dual-encoder fix: pure_style_encoder normally only ever
# sees supcon's gradient (s_pure.detach() before entering style_combiner --
# see t2sm.py _style_embeddings). The 4-seed dualenc result showed SRA_5
# sitting a real ~4.6pt below the StyleMLP baseline (z=-3.43, see memory:
# style-salad-dualenc-seed-distribution) even though style_combiner isn't
# diluting s_pure (analyze_combiner_weights.py ruled that out). One
# remaining difference from the baseline: the baseline's single StyleMLP
# encoder is trained by BOTH diffusion loss and supcon on the same weights;
# pure_style_encoder here only gets supcon. This run sets
# detach_from_generation: false (configs/reruns/ours_train_stattn_dualenc_nodetach.yaml)
# so the diffusion loss's gradient also reaches pure_style_encoder, matching
# the baseline's training regime, to test whether that closes the gap.
# Seed42 first as a directional check -- if promising, needs the other 3
# seeds before drawing a conclusion (see memory: a single-seed z-score
# against a 4-seed baseline was misleading once already in this project).
# Non-destructive: writes to checkpoints/t2sm/reruns/ours_train_stattn_dualenc_nodetach/ (gitignored).
set -uo pipefail

REPO_ROOT="/mnt/sda/xyh/style-salad"
VENV="${REPO_ROOT}/.venv"
CONFIG="configs/reruns/ours_train_stattn_dualenc_nodetach.yaml"
LOG_DIR="${REPO_ROOT}/artifacts/run_logs"
TRAIN_LOG="${LOG_DIR}/train_stattn_dualenc_nodetach.log"
EVAL_LOG="${LOG_DIR}/evaluate_stattn_dualenc_nodetach.log"
STATUS_FILE="${LOG_DIR}/STATUS_stattn_dualenc_nodetach"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -Is)] START train (StyleSTAttn + dual pure_style_encoder, detach_from_generation=false, seed=42, GPU0)" | tee -a "${STATUS_FILE}"
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
  --csv_name metrics_ours_train_stattn_dualenc_nodetach.csv \
  > "${EVAL_LOG}" 2>&1
eval_rc=$?
echo "[$(date -Is)] evaluate exit_code=${eval_rc}" | tee -a "${STATUS_FILE}"

if [ "${eval_rc}" -eq 0 ]; then
  echo "[$(date -Is)] DONE: train+evaluate finished successfully" | tee -a "${STATUS_FILE}"
else
  echo "[$(date -Is)] DONE: evaluate failed" | tee -a "${STATUS_FILE}"
fi
exit "${eval_rc}"
