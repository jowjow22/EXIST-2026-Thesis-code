"""
ablation_harness.py
-------------------
Evaluation harness for EXIST 2026 video sexism detection.
 
Runs the same trained model under different ablation conditions by toggling
forward() flags and re-running compute_metrics. No retraining required —
each condition uses the same checkpoint.
 
Conditions:
    B0   — content only      (use_bio=False, use_viewer_emb=False, use_tau=False)
    B1   — bio, no viewer    (use_bio=True,  use_viewer_emb=False, use_tau=False)
    B2   — bio + viewer      (use_bio=True,  use_viewer_emb=True,  use_tau=False)
    FULL — bio + viewer + τ  (use_bio=True,  use_viewer_emb=True,  use_tau=True)
 
USAGE (after the main training script has saved checkpoints):
 
    from ablation_harness import run_all_ablations
    results = run_all_ablations(model, val_dataset, compute_metrics, device="cuda")
    print_ablation_table(results)
"""
 
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import OrderedDict
 
 
# ──────────────────────────────────────────────────────────────────────────
# Conditions registry
# ──────────────────────────────────────────────────────────────────────────
ABLATION_CONDITIONS = OrderedDict([
    ("B0_content_only",       {"use_bio": False, "use_viewer_emb": False}),
    ("B1_bio_no_viewer",      {"use_bio": True,  "use_viewer_emb": False}),
    ("B2_bio_viewer_no_tau",  {"use_bio": True,  "use_viewer_emb": True}),
    ("FULL_bio_viewer_tau",   {"use_bio": True,  "use_viewer_emb": True}),
])
 
# Metrics we care about for the final paper table (subset of PyEvALL output)
PAPER_METRICS = [
    "task_a_icmsoftnorm_soft",
    "task_a_icmnorm_hard",
    "task_a_fmeasure_hard",
    "task_b_icmsoftnorm_soft",
    "task_b_icmnorm_hard",
    "task_c_icmsoftnorm_soft",
    "task_c_icmnorm_hard",
]
 
 
# ──────────────────────────────────────────────────────────────────────────
# Custom collate — preserves all the non-standard fields
# ──────────────────────────────────────────────────────────────────────────
def ablation_collate(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        # scalars / 0-dim tensors → stack as 1-d
        if vals[0].dim() == 0:
            out[k] = torch.stack(vals)
        else:
            out[k] = torch.stack(vals)
    return out
 
 
# ──────────────────────────────────────────────────────────────────────────
# Run inference under one condition
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataset, flags, device, batch_size=4):
    """Run the model on the full eval set under one set of ablation flags."""
    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=ablation_collate, num_workers=0,
    )

    all_logits_a, all_logits_b, all_logits_c = [], [], []
    all_labels_a, all_labels_b, all_labels_c = [], [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        labels_a = batch.pop("labels_a")
        labels_b = batch.pop("labels_b")
        labels_c = batch.pop("labels_c")

        out = model(**batch, **flags)

        # ── FIX: handle tuple return from forward ─────────────────────
        # forward() returns (logits_a, logits_b, logits_c) at inference time,
        # or (loss, logits_a, logits_b, logits_c) at training time.
        # We popped the labels above so loss is None and forward returns the 3-tuple.
        if len(out) == 3:
            logits_a, logits_b, logits_c = out
        elif len(out) == 4:
            _loss, logits_a, logits_b, logits_c = out
        else:
            raise RuntimeError(f"Unexpected forward return length: {len(out)}")

        all_logits_a.append(logits_a.float().cpu().numpy())
        all_logits_b.append(logits_b.float().cpu().numpy())
        all_logits_c.append(logits_c.float().cpu().numpy())
        all_labels_a.append(labels_a.cpu().numpy())
        all_labels_b.append(labels_b.cpu().numpy())
        all_labels_c.append(labels_c.cpu().numpy())

    logits_a = np.concatenate(all_logits_a, axis=0)
    logits_b = np.concatenate(all_logits_b, axis=0)
    logits_c = np.concatenate(all_logits_c, axis=0)
    labels_a = np.concatenate(all_labels_a, axis=0)
    labels_b = np.concatenate(all_labels_b, axis=0)
    labels_c = np.concatenate(all_labels_c, axis=0)

    class FakeEvalPred:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids

    return FakeEvalPred(
        predictions=(logits_a, logits_b, logits_c),
        label_ids=(labels_a, labels_b, labels_c),
    )
# ──────────────────────────────────────────────────────────────────────────
# Run all conditions
# ──────────────────────────────────────────────────────────────────────────
def run_all_ablations(model, dataset, compute_metrics_fn, device="cuda",
                      batch_size=4, conditions=None):
    """Evaluate the model under every condition.
 
    Args:
        model:               EXISTVideoModel instance (trained, in eval mode)
        dataset:             validation or test set (EXISTVideoDataset)
        compute_metrics_fn:  the compute_metrics function from training script
        device:              "cuda" or "cpu"
        batch_size:          inference batch size
        conditions:          dict of {name: flags}, defaults to ABLATION_CONDITIONS
 
    Returns:
        dict {condition_name: metrics_dict}
    """
    if conditions is None:
        conditions = ABLATION_CONDITIONS
 
    results = OrderedDict()
    for name, flags in conditions.items():
        print(f"\n─── Running condition: {name} ─── flags={flags}")
        eval_pred = run_inference(model, dataset, flags, device, batch_size)
        metrics   = compute_metrics_fn(eval_pred)
        results[name] = metrics
        # Quick summary line
        a_soft = metrics.get("task_a_icmsoftnorm_soft", float("nan"))
        c_soft = metrics.get("task_c_icmsoftnorm_soft", float("nan"))
        print(f"   Task A ICM-Soft-Norm = {a_soft:.4f} | Task C ICM-Soft-Norm = {c_soft:.4f}")
 
    return results
 
 
# ──────────────────────────────────────────────────────────────────────────
# Statistical robustness: re-run with different random seeds for the
# stochastic components (currently none — model is deterministic at eval
# time — but keep the hook for when you add dropout-at-eval or MC dropout)
# ──────────────────────────────────────────────────────────────────────────
def run_ablations_multi_seed(model_fn, dataset, compute_metrics_fn,
                              seeds=(42, 7, 2025), device="cuda"):
    """Run ablations over multiple seeds. model_fn(seed) returns a loaded model.
 
    Use this once you train multiple checkpoints with different seeds.
    Returns a dict {condition: {metric: [val_seed0, val_seed1, ...]}}.
    """
    aggregated = {c: {m: [] for m in PAPER_METRICS} for c in ABLATION_CONDITIONS}
 
    for seed in seeds:
        print(f"\n══════ Seed {seed} ══════")
        model = model_fn(seed)
        results = run_all_ablations(model, dataset, compute_metrics_fn, device)
        for condition, metrics in results.items():
            for m in PAPER_METRICS:
                if m in metrics:
                    aggregated[condition][m].append(metrics[m])
 
    # Aggregate mean ± std
    summary = {}
    for condition, metric_lists in aggregated.items():
        summary[condition] = {}
        for m, vals in metric_lists.items():
            if vals:
                summary[condition][m] = (np.mean(vals), np.std(vals))
            else:
                summary[condition][m] = (float("nan"), float("nan"))
    return summary
 
 
# ──────────────────────────────────────────────────────────────────────────
# Pretty printing — single seed
# ──────────────────────────────────────────────────────────────────────────
def print_ablation_table(results, metrics=None):
    """Print a clean ablation table to stdout and return it as a DataFrame.
 
    results: output of run_all_ablations
    """
    if metrics is None:
        metrics = PAPER_METRICS
 
    rows = []
    for condition, metric_dict in results.items():
        row = {"Condition": condition}
        for m in metrics:
            row[m] = metric_dict.get(m, float("nan"))
        rows.append(row)
 
    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("ABLATION RESULTS")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
 
    # Deltas relative to B0 (content only) — this is the publishable headline
    print("\n" + "─" * 100)
    print("DELTAS vs B0 (content only)")
    print("─" * 100)
    if "B0_content_only" in results:
        b0 = results["B0_content_only"]
        delta_rows = []
        for condition in results:
            if condition == "B0_content_only":
                continue
            row = {"Condition": condition}
            for m in metrics:
                if m in results[condition] and m in b0:
                    row[m] = results[condition][m] - b0[m]
                else:
                    row[m] = float("nan")
            delta_rows.append(row)
        delta_df = pd.DataFrame(delta_rows)
        print(delta_df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
 
    return df
 
 
# ──────────────────────────────────────────────────────────────────────────
# Pretty printing — multi-seed with mean ± std
# ──────────────────────────────────────────────────────────────────────────
def print_multi_seed_table(summary, metrics=None):
    """Print a publishable-format table with mean ± std across seeds."""
    if metrics is None:
        metrics = PAPER_METRICS
 
    rows = []
    for condition, metric_dict in summary.items():
        row = {"Condition": condition}
        for m in metrics:
            mean, std = metric_dict[m]
            row[m] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)
 
    df = pd.DataFrame(rows)
    print("\n" + "=" * 120)
    print("ABLATION RESULTS (mean ± std over seeds)")
    print("=" * 120)
    print(df.to_string(index=False))
    return df
 
 
# ──────────────────────────────────────────────────────────────────────────
# Save to CSV/LaTeX for paper inclusion
# ──────────────────────────────────────────────────────────────────────────
def save_for_paper(df, csv_path="ablation_results.csv", latex_path="ablation_results.tex"):
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV to {csv_path}")
    try:
        latex = df.to_latex(index=False, escape=True, float_format="%.3f")
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"✓ Saved LaTeX to {latex_path}")
    except Exception as e:
        print(f"  ! LaTeX export failed: {e}")
 
 
# ──────────────────────────────────────────────────────────────────────────
# Minimal usage example (run AFTER training completes)
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Example usage:
 
        # 1. Train your model normally (full FULL config, all flags True)
        trainer.train()
 
        # 2. Load best checkpoint
        from transformers import Trainer
        model = trainer.model  # already loaded best at end
 
        # 3. Run ablations
        from ablation_harness import (
            run_all_ablations, print_ablation_table, save_for_paper
        )
        results = run_all_ablations(model, val_dataset, compute_metrics, device="cuda")
        df = print_ablation_table(results)
        save_for_paper(df)
    """
    print("Import this module and call run_all_ablations(model, dataset, compute_metrics).")
 