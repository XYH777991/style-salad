#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python -m style_salad.cli.generate \
  --config "${CONFIG:-configs/ours.yaml}" \
  --ref_motion_id "${REF_MOTION_ID:-030273}" \
  --caption "${CAPTION:-a person walks forward}" \
  --num_samples "${NUM_SAMPLES:-8}" \
  --output_length "${OUTPUT_LENGTH:-196}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS:-50}" \
  "$@"
