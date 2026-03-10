#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <lm_model_path> <rm_model_path> <output_dir> <num_procs>"
  exit 1
fi

LM_MODEL_PATH="$1"
RM_MODEL_PATH="$2"
OUTPUT_DIR="$3"
NUM_PROCS="$4"

VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-7}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
VLLM_URL="http://127.0.0.1:${VLLM_PORT}"


mkdir -p "${OUTPUT_DIR}"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES}" \
nohup python -m vllm.entrypoints.openai.api_server \
  --model "${LM_MODEL_PATH}" \
  --dtype bfloat16 \
  --tensor-parallel-size "${VLLM_TP_SIZE}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" > "${OUTPUT_DIR}/vllm.log" 2>&1 &
VLLM_PID=$!

until curl -sf "${VLLM_URL}/v1/models" >/dev/null; do sleep 1; done

bash scripts/qwen_ins_aime24.sh "${LM_MODEL_PATH}" "${RM_MODEL_PATH}" "${OUTPUT_DIR}" "${NUM_PROCS}"
