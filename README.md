# $\nabla$-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space

The official implementation of the ICLR 2026 paper [$\nabla$-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](https://arxiv.org/abs/2603.04948).

[Peihao Wang](https://peihaowang.github.io/)*, [Ruisi Cai](https://cairuisi.github.io/)*, [Zhen Wang](https://zhenwang9102.github.io/), [Hongyuan Mei](https://www.hongyuanmei.com/), [Qiang Liu](https://www.cs.utexas.edu/~lqiang/), [Pan Li](https://sites.google.com/view/panli-purdue/home), [Atlas Wang](https://vita-group.github.io/research.html)

*International Conference on Learning Representations (ICLR), 2026*  
\* denotes equal contribution.

[Paper](https://arxiv.org/abs/2603.04948) | [Code](https://github.com/VITA-Group/Nabla-Reasoner)

## Get Started

### Environment

We tested this release with the environment below (nearby versions should also work).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install packaging lightning==2.5.0 lightning[app] lightning[data] rich
pip install transformers==4.56.1 tokenizers==0.22.0 datasets==4.0.0 accelerate==1.10.1
pip install jsonargparse[signatures] sentencepiece wandb torchmetrics psutil
pip install tensorboard zstandard pandas pyarrow huggingface_hub
pip install flash-attn==2.8.3
pip install einops opt_einsum
pip install latex2sympy2 word2number pylatexenc
pip install vllm==0.10.2
```

### Data

Evaluation sets are adapted from [Spurious_Rewards](https://github.com/ruixin31/Spurious_Rewards/tree/main/code/data) and hosted at [peihaowang/math-reasoning-eval](https://huggingface.co/datasets/peihaowang/math-reasoning-eval).

Built-in benchmarks:

- `AMC`
- `AIME-2024`
- `AIME-2025`
- `MATH-500`

## Usage

### Launch vLLM

Start a vLLM server before running $\nabla$-Reasoner with `backend="vllm"`.

```bash
python -m vllm.entrypoints.openai.api_server \
  --model <lm_model_path> \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --port 8000
```

### Python API Example

The snippet below shows the minimal flow: initialize models, configure optimization hyperparameters, create `NablaDecoding`, then generate a response.

```python
import torch
from decoding import NablaDecoding

device = "cuda:0"

# Initialize base and rewards models and their tokenizers
# lm_model, lm_tokenizer, rm_model, rm_tokenizer = ...

train_args = {
    "max_iters": 100,
    "warmup_iters_ratio": 0.0,
    "learning_rate": 0.01,
    "min_lr_ratio": 0.1,
    "weight_decay": 0.0,
    "reward_coeff": 1.0,
    "mixed_precision": torch.bfloat16,
    "grad_caching": True,
    "update_postfix": False,
    "embedder_type": "latents",
}

decoder = NablaDecoding(
    lm_model,
    lm_tokenizer,
    rm_model,
    rm_tokenizer,
    train_args,
    device=device,
    max_length=3072,
    verbose=2,
    rejection_sampling=True,
    max_n_generations=8,
    rollout_tau=0.7,
    rollout_top_k=20,
    rollout_top_p=0.8,
    resample_tau=0.5,
    resample_top_k=20,
    resample_top_p=0.8,
    backend="vllm",
    vllm_url="http://127.0.0.1:8000",
    vllm_model_name=lm_model_path,
    confidence_selector_threshold=0.97,
    grad_selector_threshold=8,
)

prompt = "Solve: If 2x + 3 = 11, what is x?"
token_ids = decoder.generate(prompt, return_prompt=False, seed=42)
response = lm_tokenizer.decode(token_ids[0], skip_special_tokens=True)
```

If your vLLM version does not support token-id based I/O (for example `vllm <= 0.7.2`), set `vllm_output_type="text"`. You can also run with `backend="huggingface"` (no vLLM server required).

### Single-Prompt CLI

Use `run.py` for quick test on one prompts.

```bash
python run.py \
  --lm_model_name <lm_model_path> \
  --rm_model_name <rm_model_path> \
  --vllm_url http://127.0.0.1:8000 \
  --vllm_model_name <lm_model_path> \
  --prompt "<your prompt>" \
  --embedder_type latents \
  --max_iters 100 \
  --learning_rate 0.01 \
  --reward_coeff 1.0
```

### Parallel Benchmark Run

Use `multi_run.py` to process an entire benchmark in parallel across multiple workers/GPUs.

```bash
python multi_run.py \
  --lm_model_name <lm_model_path> \
  --rm_model_name <rm_model_path> \
  --vllm_url http://127.0.0.1:8000 \
  --vllm_model_name <lm_model_path> \
  --prompts <dataset_name> \
  --num_procs 8 \
  --output_dir <output_dir> \
  --embedder_type latents \
  --n_generations 8 \
  --max_iters 100 \
  --learning_rate 0.01 \
  --reward_coeff 1.0
```

### Evaluation

After generation, compute benchmark metrics from `responses.json` using:

```bash
python eval_outputs.py \
  --json_path <output_dir>/responses.json \
  --output_file <output_dir>/eval.json
```

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{wang2026nabla,
  title={Nabla-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Textual Space},
  author={Wang, Peihao and Cai, Ruisi and Wang, Zhen and Mei, Hongyuan and Liu, Qiang and Li, Pan and Wang, Atlas},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
