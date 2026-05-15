"""
Layer-wise linear probing for LoRA placement (Method #3 from
docs/future_layer_selection_methods.md).

Asks: does a non-gradient one-shot signal (per-layer probe accuracy) escape
the capacity-controller trap that SNIP/GN/Fisher fell into at layer
granularity? See docs/post_arxiv_work_plan.md §W2.

Pipeline (per (task, seed) cell):
  1. Freeze roberta-base. Forward 5% of the training set with
     output_hidden_states=True. Extract CLS-pooled activations at each
     of the 12 transformer-block outputs.
  2. Fit a linear probe per layer:
       - LogisticRegression for classification
       - Ridge regression for STS-B (regression)
     Evaluate probe on the held-out validation split.
  3. Select top-N layers by probe accuracy under one of three rules:
       low      — bottom-N (features not yet present; LoRA should build them)
       mid      — middle-N (features forming; LoRA should refine them)
       high     — top-N    (features present; small updates have max leverage)
     Map each selected layer to its 6 standard LoRA modules
     (q, k, v, attention-out, ffn_up, ffn_down).
  4. Train rank-8 PEFT-LoRA on the selected modules with the same
     hyperparameters used by random_lora.py and the SG-LoRA scripts.

N is round(top_k_percent * 12). Default top_k_percent=0.20 → 2 layers
(12 modules / 16.7% of 72). This matches the parent paper's RQ3 / RQ6
focal point for direct comparison against the Random control.

W&B projects: {TASK}-Probe-{Low|Mid|High}-LoRA-5-Seeds-2

Usage:
  python probe_lora.py --task rte  --rule high --smoke
  python probe_lora.py --task rte  --rule high
  python probe_lora.py --task rte  --rule low  --seeds 15 25 35 45 55
  python probe_lora.py --task rte  --rule high --resume
"""

import argparse
import os
import random
import shutil
import tempfile
import time
from typing import Dict, List

import numpy as np
import torch
import wandb
import evaluate
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Per-task config (matches random_lora.py and the SG-LoRA scripts).
DATASET_CONFIGS = {
    "sst2": {"sentence_keys": ["sentence"],               "num_labels": 2, "metric_to_optimize": "accuracy",              "eval_split": "validation"},
    "mrpc": {"sentence_keys": ["sentence1", "sentence2"], "num_labels": 2, "metric_to_optimize": "accuracy",              "eval_split": "validation"},
    "cola": {"sentence_keys": ["sentence"],               "num_labels": 2, "metric_to_optimize": "matthews_correlation", "eval_split": "validation"},
    "stsb": {"sentence_keys": ["sentence1", "sentence2"], "num_labels": 1, "metric_to_optimize": "pearson",               "eval_split": "validation"},
    "rte":  {"sentence_keys": ["sentence1", "sentence2"], "num_labels": 2, "metric_to_optimize": "accuracy",              "eval_split": "validation"},
}

TASK_HPARAMS = {
    "sst2": {"learning_rate": 3e-4, "num_train_epochs": 10, "eval_steps": 100, "logging_steps": 50},
    "mrpc": {"learning_rate": 3e-4, "num_train_epochs": 15, "eval_steps": 20,  "logging_steps": 10},
    "cola": {"learning_rate": 2e-4, "num_train_epochs": 20, "eval_steps": 50,  "logging_steps": 25},
    "stsb": {"learning_rate": 3e-4, "num_train_epochs": 15, "eval_steps": 50,  "logging_steps": 25},
    "rte":  {"learning_rate": 3e-4, "num_train_epochs": 20, "eval_steps": 20,  "logging_steps": 10},
}

MODEL_NAME = "roberta-base"
NUM_LAYERS = 12
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
MAX_SEQ_LENGTH = 128
TOP_K_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
SEEDS_TO_RUN = [15, 25, 35, 45, 55]
PROBE_BATCH_SIZE = 32
PERCENT_PROBE_SAMPLES = 0.05  # match parent paper's saliency budget
CHECKPOINT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".probe_lora_ckpt")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ----- Probe scoring -----------------------------------------------------

def extract_layer_activations(model, dataloader, device: str) -> List[np.ndarray]:
    """Forward each batch through the frozen backbone with output_hidden_states=True
    and collect CLS-pooled activations at every encoder layer. Returns a list of
    length NUM_LAYERS, each entry an array of shape (N, hidden_dim).
    """
    model.eval().to(device)
    per_layer: List[List[np.ndarray]] = [[] for _ in range(NUM_LAYERS)]
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            # out.hidden_states is a tuple of (NUM_LAYERS+1) tensors.
            # Index 0 = embeddings, 1..NUM_LAYERS = encoder layer outputs.
            for L in range(NUM_LAYERS):
                cls = out.hidden_states[L + 1][:, 0, :].cpu().float().numpy()
                per_layer[L].append(cls)
    return [np.concatenate(parts, axis=0) for parts in per_layer]


def fit_layer_probes(train_acts: List[np.ndarray], train_y: np.ndarray,
                      eval_acts: List[np.ndarray], eval_y: np.ndarray,
                      task: str, seed: int) -> Dict[int, float]:
    """Fit one linear probe per layer, return {layer_idx: probe_score}."""
    is_regression = DATASET_CONFIGS[task]["num_labels"] == 1
    scores: Dict[int, float] = {}
    for L in range(NUM_LAYERS):
        Xtr, Xev = train_acts[L], eval_acts[L]
        if is_regression:
            probe = Ridge(alpha=1.0, random_state=seed)
            probe.fit(Xtr, train_y)
            preds = probe.predict(Xev)
            r, _ = pearsonr(preds, eval_y)
            scores[L] = float(r) if not np.isnan(r) else 0.0
        else:
            probe = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1,
                                        random_state=seed,
                                        multi_class="auto", solver="lbfgs")
            probe.fit(Xtr, train_y)
            preds = probe.predict(Xev)
            scores[L] = float(accuracy_score(eval_y, preds))
    return scores


def calculate_probe_scores(dataset_name: str, percent_samples: float,
                            seed: int, is_smoke_test: bool = False) -> Dict[int, float]:
    """Full probe pipeline for one (task, seed): tokenize, forward through
    frozen backbone, fit per-layer probes, return per-layer accuracy/Pearson.
    """
    set_seed(seed)
    d_cfg = DATASET_CONFIGS[dataset_name]
    keys = d_cfg["sentence_keys"]
    is_regression = d_cfg["num_labels"] == 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    raw = load_dataset("glue", dataset_name)
    eval_split = d_cfg["eval_split"]

    n_total = len(raw["train"])
    n_probe = 64 if is_smoke_test else max(64, int(n_total * percent_samples))
    train_subset = raw["train"].shuffle(seed=seed).select(range(min(n_probe, n_total)))

    n_eval = min(512, len(raw[eval_split])) if is_smoke_test else len(raw[eval_split])
    eval_subset = raw[eval_split].select(range(n_eval))

    def collate(batch):
        args = [[ex[k] for ex in batch] for k in keys]
        enc = tokenizer(*args, truncation=True, padding="max_length",
                         max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        return enc

    def labels_from(ds):
        if is_regression:
            return np.array([float(ex["label"]) for ex in ds], dtype=np.float32)
        return np.array([int(ex["label"]) for ex in ds], dtype=np.int64)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=PROBE_BATCH_SIZE, shuffle=False, collate_fn=collate)
    eval_loader = torch.utils.data.DataLoader(
        eval_subset, batch_size=PROBE_BATCH_SIZE, shuffle=False, collate_fn=collate)

    # Use AutoModel (no classifier head) for activation extraction. The
    # encoder weights are bit-identical to those of AutoModelForSequenceClassification
    # before fine-tuning.
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_acts = extract_layer_activations(backbone, train_loader, device)
    eval_acts  = extract_layer_activations(backbone, eval_loader,  device)
    backbone.cpu()
    del backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_y = labels_from(train_subset)
    eval_y  = labels_from(eval_subset)
    scores = fit_layer_probes(train_acts, train_y, eval_acts, eval_y,
                               task=dataset_name, seed=seed)
    return scores


# ----- Selection rules ---------------------------------------------------

def select_layers_by_probe(probe_scores: Dict[int, float], top_k_percent: float,
                            rule: str) -> List[int]:
    """Return list of selected layer indices under one of three rules.
    `low`  — bottom N by probe score
    `high` — top    N by probe score
    `mid`  — middle N by probe score (centered on the median, ties broken by lower index)
    N = max(1, round(top_k_percent * NUM_LAYERS)).
    """
    n_select = max(1, round(top_k_percent * NUM_LAYERS))
    ordered = sorted(probe_scores.items(), key=lambda kv: (kv[1], kv[0]))  # ascending
    if rule == "low":
        picked = ordered[:n_select]
    elif rule == "high":
        picked = ordered[-n_select:]
    elif rule == "mid":
        start = (NUM_LAYERS - n_select) // 2
        picked = ordered[start:start + n_select]
    else:
        raise ValueError(f"Unknown rule {rule!r}")
    return sorted(L for L, _ in picked)


def expand_layers_to_modules(selected_layers: List[int]) -> List[str]:
    """Map each selected layer to its 6 standard LoRA-target modules
    (q, k, v, attention-output, ffn-up, ffn-down). Mirrors the 72-candidate
    pool that random_lora.py and the SG-LoRA scripts rank over.
    """
    modules: List[str] = []
    for L in selected_layers:
        modules += [
            f"roberta.encoder.layer.{L}.attention.self.query",
            f"roberta.encoder.layer.{L}.attention.self.key",
            f"roberta.encoder.layer.{L}.attention.self.value",
            f"roberta.encoder.layer.{L}.attention.output.dense",
            f"roberta.encoder.layer.{L}.intermediate.dense",
            f"roberta.encoder.layer.{L}.output.dense",
        ]
    return modules


# ----- W&B resume helper -------------------------------------------------

def fetch_completed_runs(task: str, rule: str) -> set:
    api = wandb.Api()
    project = f"{task.upper()}-Probe-{rule.capitalize()}-LoRA-5-Seeds-2"
    try:
        entity = api.viewer.entity
        runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})
    except Exception as e:
        print(f"[resume] WARNING: could not query W&B project {project}: {e}")
        return set()
    done = set()
    for r in runs:
        s = r.config.get("seed"); k = r.config.get("top_k_percent")
        if s is not None and k is not None:
            done.add((int(s), round(float(k), 2)))
    print(f"[resume] {len(done)} already-finished runs in {project}")
    return done


def _find_last_checkpoint(ckpt_dir: str):
    import glob
    cps = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint-*")))
    return cps[-1] if cps else None


# ----- Fine-tuning -------------------------------------------------------

def fine_tune_model(cfg: dict, model_to_train, run_name: str, ckpt_dir: str) -> None:
    is_smoke = cfg.get("is_smoke_test", False)
    d_cfg = DATASET_CONFIGS[cfg["dataset_name"]]
    keys = d_cfg["sentence_keys"]; eval_split = d_cfg["eval_split"]
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_from = None if is_smoke else _find_last_checkpoint(ckpt_dir)
    if resume_from:
        print(f"[checkpoint] resuming from {resume_from}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    dataset = load_dataset("glue", cfg["dataset_name"])

    def preprocess(examples):
        args = [examples[k] for k in keys]
        return tokenizer(*args, truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)

    cols_remove = list(keys)
    if "idx" in dataset["train"].column_names:
        cols_remove.append("idx")
    tok = dataset.map(preprocess, batched=True, remove_columns=cols_remove)
    tok = tok.rename_column("label", "labels")
    tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    if is_smoke:
        tok["train"] = tok["train"].select(range(min(128, len(tok["train"]))))
        tok[eval_split] = tok[eval_split].select(range(min(64, len(tok[eval_split]))))

    metric = evaluate.load("glue", cfg["dataset_name"])

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.squeeze(preds) if d_cfg["num_labels"] == 1 else np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    targs = TrainingArguments(
        output_dir=ckpt_dir, run_name=run_name,
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        fp16=True, num_train_epochs=cfg["num_train_epochs"],
        weight_decay=cfg["weight_decay"], load_best_model_at_end=True,
        metric_for_best_model=d_cfg["metric_to_optimize"], greater_is_better=True,
        report_to="wandb", push_to_hub=False,
        logging_strategy="steps", eval_strategy="steps", save_strategy="steps",
        logging_steps=cfg["logging_steps"], eval_steps=cfg["eval_steps"],
        save_steps=cfg["eval_steps"], save_total_limit=2,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 2),
        max_steps=cfg.get("max_steps", -1), seed=cfg["seed"],
    )
    trainer = Trainer(model=model_to_train, args=targs,
                      train_dataset=tok["train"], eval_dataset=tok[eval_split],
                      tokenizer=tokenizer, compute_metrics=compute_metrics)
    trainer.train(resume_from_checkpoint=resume_from)
    eval_results = trainer.evaluate(metric_key_prefix="eval_primary")
    wandb.summary.update({f"best_{k}": v for k, v in eval_results.items()})

    if not is_smoke:
        with tempfile.TemporaryDirectory() as tmp_art:
            trainer.save_model(tmp_art)
            art = wandb.Artifact(name=f"{run_name.split('-seed')[0]}-best-model", type="model")
            art.add_dir(tmp_art)
            wandb.log_artifact(art)
    shutil.rmtree(ckpt_dir, ignore_errors=True)


# ----- Single experiment cell --------------------------------------------

def run_single_experiment(cfg: dict) -> None:
    project = f"{cfg['dataset_name'].upper()}-Probe-{cfg['rule'].capitalize()}-LoRA-5-Seeds-2"
    wandb.init(project=project, config=cfg, name=cfg["run_name"], group=cfg["group"], save_code=True)
    set_seed(cfg["seed"])

    num_labels = DATASET_CONFIGS[cfg["dataset_name"]]["num_labels"]

    # Step 1: per-layer probe accuracy.
    start = time.time()
    probe_scores = calculate_probe_scores(
        cfg["dataset_name"], PERCENT_PROBE_SAMPLES, cfg["seed"],
        is_smoke_test=cfg.get("is_smoke_test", False))
    wandb.log({"time/saliency_calculation_seconds": time.time() - start})

    # Log the full per-layer probe table.
    probe_table = wandb.Table(columns=["Layer", "Probe score"])
    for L in range(NUM_LAYERS):
        probe_table.add_data(L, float(probe_scores[L]))
    wandb.log({"probe_scores": probe_table})
    wandb.summary.update({f"probe_score_layer_{L}": float(probe_scores[L]) for L in range(NUM_LAYERS)})

    # Step 2: select layers under the chosen rule.
    selected_layers = select_layers_by_probe(probe_scores, cfg["top_k_percent"], cfg["rule"])
    target_modules = expand_layers_to_modules(selected_layers)
    print(f"[probe-{cfg['rule']}] selected layers: {selected_layers}  "
          f"-> {len(target_modules)} modules")

    sel_table = wandb.Table(columns=["Selected Layer", "Probe score"])
    for L in selected_layers:
        sel_table.add_data(L, float(probe_scores[L]))
    wandb.log({"selected_layers": sel_table})

    # Step 3: PEFT LoRA on the selected modules.
    base = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=num_labels)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=target_modules, bias="none",
    )
    model = get_peft_model(base, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    wandb.config.update({
        "trainable_params": trainable, "all_params": total,
        "trainable_percent": (trainable / total) * 100.0,
        "n_selected_layers": len(selected_layers),
        "n_selected_modules": len(target_modules),
        "selected_layer_indices": selected_layers,
    })

    # Step 4: train.
    ckpt_dir = os.path.join(CHECKPOINT_BASE, cfg["run_name"])
    start = time.time()
    fine_tune_model(cfg, model, cfg["run_name"], ckpt_dir)
    wandb.log({"time/training_seconds": time.time() - start})

    wandb.finish()
    del model, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_study(task: str, seed: int, rule: str, top_k_percents: List[float],
               is_smoke: bool = False, skip_completed: set = None) -> None:
    hp = TASK_HPARAMS[task]
    base_cfg = {
        "model_name": MODEL_NAME, "dataset_name": task,
        "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "seed": seed,
        "learning_rate": hp["learning_rate"], "num_train_epochs": hp["num_train_epochs"],
        "eval_steps": hp["eval_steps"], "logging_steps": hp["logging_steps"],
        "is_smoke_test": is_smoke, "dataloader_num_workers": 2,
        "rule": rule, "ablation_type": f"PROBE_{rule.upper()}",
    }
    prefix = task.upper()
    rule_disp = rule.capitalize()
    exps = []
    if is_smoke:
        s = {**base_cfg, "top_k_percent": 0.20}
        s.update({"max_steps": 8, "num_train_epochs": 1, "logging_steps": 4, "eval_steps": 4})
        s["run_name"] = f"{prefix}-Probe-{rule_disp}-p20-smoke"
        s["group"] = f"{prefix}-Probe-{rule_disp}-p20"
        exps.append(s)
    else:
        for p in top_k_percents:
            c = {**base_cfg, "top_k_percent": p}
            pct = int(p * 100)
            c["run_name"] = f"{prefix}-Probe-{rule_disp}-p{pct}-seed{seed}"
            c["group"] = f"{prefix}-Probe-{rule_disp}-p{pct}"
            exps.append(c)
    if skip_completed:
        before = len(exps)
        exps = [c for c in exps
                if (c["seed"], round(float(c["top_k_percent"]), 2)) not in skip_completed]
        if before - len(exps):
            print(f"[resume] Skipping {before - len(exps)} finished runs for seed={seed}")
    for i, c in enumerate(exps):
        print(f"\n{'='*80}\nRUN {i+1}/{len(exps)}: {c['run_name']}\n{'='*80}")
        run_single_experiment(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(DATASET_CONFIGS))
    ap.add_argument("--rule", required=True, choices=["low", "mid", "high"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_TO_RUN)
    ap.add_argument("--top-k", type=float, nargs="+", default=[0.20],
                    help="One or more top_k_percent values. Default: [0.20].")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        print(f"\n[SMOKE TEST] task={args.task}  rule={args.rule}\n")
        run_study(args.task, seed=42, rule=args.rule,
                  top_k_percents=[0.20], is_smoke=True)
        return

    skip = fetch_completed_runs(args.task, args.rule) if args.resume else None
    print(f"\n[FULL RUN] task={args.task}  rule={args.rule}  "
          f"seeds={args.seeds}  top_k={args.top_k}"
          + ("  (resuming)" if args.resume else "") + "\n")
    for seed in args.seeds:
        print(f"\n{'#'*80}\n#  seed={seed}\n{'#'*80}")
        run_study(args.task, seed=seed, rule=args.rule,
                   top_k_percents=args.top_k, is_smoke=False, skip_completed=skip)


if __name__ == "__main__":
    main()
