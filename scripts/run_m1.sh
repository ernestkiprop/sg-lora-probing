#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export WANDB_CACHE_DIR=/tmp/wandb_cache
export HF_HOME=/tmp/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /workspace/sg-lora-probing
for rule in low mid high; do for task in mrpc cola; do python scripts/probe_lora.py --task $task --rule $rule --resume; done; done
for rule in low mid high; do python scripts/probe_lora.py --task sst2 --rule $rule --seeds 15 25 --resume; done
