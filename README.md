# SG-LoRA Probing Follow-up

Layer-wise linear probing as a placement criterion for LoRA. Companion experiment
to the paper **When Does Saliency Help LoRA? A Rigorous Study of One-Shot
Gradient Criteria for Adapter Placement**.

## What this tests

The parent paper shows that one-shot **gradient-based** layer-level saliency
(SNIP, GN, Fisher) is a capacity controller, not a placement oracle: at matched
cardinality, uniform random layer selection ties every gradient criterion across
all 5 GLUE tasks.

This repo runs the cheapest remaining open question: does a **non-gradient**
one-shot signal — per-layer linear probe accuracy — escape the same trap?

Three selection rules are tested:

- **low**:  apply LoRA to the bottom-N layers by probe accuracy (features not yet present)
- **mid**:  apply LoRA to the middle-N layers (features forming)
- **high**: apply LoRA to the top-N layers    (features present; max prediction leverage)

`N = round(top_k_percent × 12)`. Default `top_k_percent=0.20` selects 2 of 12
encoder layers → 12 modules at LoRA rank 8 = ~1M trainable parameters
(matches the parent paper's k=20% selection within ~15%).

## Decision rule

The same five evaluation criteria the parent paper proposes for the field:

1. Beats matched-cardinality **random selection** at every k where it claims to win.
2. Beats **LoRA-AllAttn at matched parameter count** on ≥3 of 5 GLUE tasks.
3. Survives **Holm–Bonferroni** correction with ≥5 seeds and Cohen's `d_z`.
4. Reports **best-k** in addition to a single k cell.
5. Validates on **at least one larger model** (RoBERTa-large minimum).

Outcomes and their reading:

- If any rule beats Random on ≥2 tasks → first non-gradient placement signal;
  publishable as a short companion paper.
- If all three rules tie Random → the no-go theorem strengthens from
  "gradient-magnitude class is dead" to "all one-shot layer-level criteria
  are dead", which justifies a v2 reframing of the parent paper.

## Pipeline

For each `(task, seed, rule)` cell:

1. Freeze `roberta-base`. Forward 5% of the training set with
   `output_hidden_states=True`. Extract CLS-pooled activations at each of the
   12 encoder-block outputs.
2. Fit a linear probe per layer (`LogisticRegression` for classification,
   `Ridge` for STS-B regression). Evaluate each probe on the validation split.
3. Select top-N layers under the chosen rule. Map each selected layer to its
   6 standard LoRA modules: `q, k, v, attention.output.dense,
   intermediate.dense, output.dense`.
4. Train PEFT-LoRA (r=8, α=16, dropout=0.1) with the same task-specific
   hyperparameters used by the parent paper's `random_lora.py`.

## Quick start

```bash
pip install -r requirements.txt

# Smoke test (~3 min: probe pass + 8 training steps)
python scripts/probe_lora.py --task rte --rule high --smoke

# Full pilot: 3 rules × 5 tasks × 5 seeds = 75 runs
for rule in low mid high; do
  for task in sst2 mrpc cola stsb rte; do
    python scripts/probe_lora.py --task $task --rule $rule
  done
done
```

## Tasks and seeds

| Task  | Metric              | Seeds                  |
|-------|---------------------|------------------------|
| SST-2 | Accuracy            | 15, 25, 35, 45, 55     |
| MRPC  | Accuracy            | 15, 25, 35, 45, 55     |
| CoLA  | Matthews correlation| 15, 25, 35, 45, 55     |
| STS-B | Pearson             | 15, 25, 35, 45, 55     |
| RTE   | Accuracy            | 15, 25, 35, 45, 55     |

## W&B logging

Each run writes to project `{TASK}-Probe-{Low|Mid|High}-LoRA-5-Seeds-2`.
Logged: per-layer probe scores (12), selected layer indices, trainable
parameter count, per-step training/eval metrics, best-checkpoint model
artifact.

## Hardware

Single GPU sufficient. Tested on NVIDIA A40 24GB and L4 24GB. fp16
mixed-precision is on by default; disable in `fine_tune_model` for
CPU/MPS.

## Citation

If you use this repo, please cite the parent paper:

```bibtex
@article{kiprop2026sglora,
  title  = {When Does Saliency Help LoRA? A Rigorous Study of One-Shot
            Gradient Criteria for Adapter Placement},
  author = {Kiprop, Ernest and Nderu, Lawrence and Karanja, Mwangi},
  year   = {2026}
}
```
