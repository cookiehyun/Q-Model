"""
TP vs FP epistemic-uncertainty (entropy) plots for MC Dropout and Deep Ensemble.
Targets: ICU admission within 24h (icu_24h) and 365-day mortality (mortality_365d).
Loads pre-computed calibrated_probs npz files (no re-inference, no .pt loading)
and plots on the TEST split, at each model's confirmed prob threshold.

Produces 4 separate PNGs under results/figures_ent/:
    uncertainty_ent_tp_fp_icu24h_deepensemble.png
    uncertainty_ent_tp_fp_icu24h_mcdropout.png
    uncertainty_ent_tp_fp_mortality365d_deepensemble.png
    uncertainty_ent_tp_fp_mortality365d_mcdropout.png
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[0]))
import config

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(config.PROJECT_ROOT, "results", "figures_ent")
os.makedirs(OUT_DIR, exist_ok=True)

EXPER_MORTALITY_DIR = os.path.join(config.PROJECT_ROOT, "experiments", "mortality")

# ============================================================
# Confirmed configs — npz uses `_icu` suffix keys for BOTH tasks
# (copy-paste artifact in cali_*_mortality365d.py; keys are correct
# per-task data, just misleadingly named).
# ============================================================
TASKS = {
    "icu24h": dict(
        title_task="ICU 24h",
        models=dict(
            **{
                "Deep Ensemble": dict(
                    thr=0.11,
                    npz=os.path.join(config.PROJECT_ROOT, "experiments", "icu24h", "deepensemble",
                                      "results", "csv", "calibrated_probs_ensemble.npz"),
                ),
                "MC Dropout": dict(
                    thr=0.13,
                    npz=os.path.join(config.PROJECT_ROOT, "experiments", "icu24h", "mcdropout",
                                      "results", "csv", "calibrated_probs_mc.npz"),
                ),
            }
        ),
    ),
    "mortality365d": dict(
        title_task="Mortality 365d",
        models=dict(
            **{
                "Deep Ensemble": dict(
                    thr=0.13,
                    npz=os.path.join(EXPER_MORTALITY_DIR, "deepensemble", "results", "csv",
                                      "calibrated_probs_ensemble_mortality365d.npz"),
                ),
                "MC Dropout": dict(
                    thr=0.13,
                    npz=os.path.join(EXPER_MORTALITY_DIR, "mcdropout", "results", "csv",
                                      "calibrated_probs_mc_mortality365d.npz"),
                ),
            }
        ),
    ),
}


def plot_tp_fp_entropy(ent, true, pred, model_label, task_label, out_path):
    assert len(ent) == len(true) == len(pred), \
        f"length mismatch: ent={len(ent)}, true={len(true)}, pred={len(pred)}"

    tp_mask = (pred == 1) & (true == 1)
    fp_mask = (pred == 1) & (true == 0)
    tp_ent, fp_ent = ent[tp_mask], ent[fp_mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, max(tp_ent.max(), fp_ent.max()), 40)
    ax.hist(tp_ent, bins=bins, alpha=0.6, color='tab:blue', label=f"TP (n={tp_mask.sum()})")
    ax.hist(fp_ent, bins=bins, alpha=0.6, color='tab:red',  label=f"FP (n={fp_mask.sum()})")
    ax.set_title(f"{model_label} (entropy) with {task_label}")
    ax.set_xlabel("Uncertainty (Entropy)", fontweight='bold')
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


for task_key, task_cfg in TASKS.items():
    for model_label, mcfg in task_cfg["models"].items():
        data = np.load(mcfg["npz"])

        test_ent  = data["test_ent_icu"]
        test_true = data["test_true_icu"]
        test_prob = data["test_prob_icu"]
        test_pred = (test_prob >= mcfg["thr"]).astype(int)

        fname_tag = model_label.lower().replace(" ", "")
        out_path = os.path.join(OUT_DIR, f"uncertainty_ent_tp_fp_{task_key}_{fname_tag}.png")
        plot_tp_fp_entropy(test_ent, test_true, test_pred, model_label, task_cfg["title_task"], out_path)

print("All entropy plots complete.")