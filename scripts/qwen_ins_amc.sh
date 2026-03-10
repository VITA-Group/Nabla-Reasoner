#!/usr/bin/env bash

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <lm_model_path> <rm_model_path> <output_dir> <num_procs>"
  exit 1
fi

LM_MODEL_PATH="$1"
RM_MODEL_PATH="$2"
OUTPUT_DIR="$3"
NUM_PROCS="$4"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python eval/multi_run.py \
  --lm_model_name "${LM_MODEL_PATH}" \
  --rm_model_name "${RM_MODEL_PATH}" \
  --vllm_url "http://127.0.0.1:8000" \
  --vllm_model_name "${LM_MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_procs "${NUM_PROCS}" \
  --prompts AMC \
  --max_generation_len 1024 \
  --attn_implementation flash_attention_2 \
  --embedder_type latents \
  --n_generations 8 \
  --rollout_tau 0.7 \
  --rollout_top_p 0.8 \
  --rollout_top_k 20 \
  --resample_tau 0.5 \
  --resample_top_p 0.8 \
  --resample_top_k 20 \
  --reward_coeff 1.0 \
  --max_iters 20 \
  --learning_rate 0.01 \
  --confidence_threshold 0.97 \
  --grad_threshold 4 \
  --verbose 2

python eval/eval_outputs.py --json_path "${OUTPUT_DIR}/responses.json" --output_file "${OUTPUT_DIR}/eval.json"

