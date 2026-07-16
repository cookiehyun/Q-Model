# Q-model

A pipeline for reducing false alarms (FPs) in clinical early-warning prediction, by training a separate **Q-model** that predicts whether a base model's prediction is **correct or incorrect**, and using that to suppress likely-wrong alarms post-hoc.

Two prediction targets are supported, sharing the same pipeline and code structure:

| Target | Description | Positive rate |
|---|---|---|
| `icu24h` | ICU admission within 24h | moderate (see Results) |
| `mortality` | 365-day mortality (`deterioration_mortality_365d`) | ~13.8% |

## 1. Overview

Clinical early-warning models are typically operated at a fixed sensitivity (recall) threshold (≥ 0.80). At this operating point, the number of false alarms (FPs) tends to be high, leading to alarm fatigue.

The Q-model addresses this problem as follows:

```
[1] Train Base Model
     └─ Input: structured patient data (demographics, vitals, labvalues, biometrics) + missingness mask
     └─ Output: predicted probability of the target (prob)

[2] Calibrate the base model's probability (fit on VAL, applied to TRAIN/TEST)
     └─ Platt Scaling and Isotonic Regression, each added as a SEPARATE Q-model
        input feature (not replacing the original probability)
     └─ Calibrators are always fit on VAL and evaluated on a disjoint split,
        never fit/eval on the same data (this previously caused Isotonic
        Regression to overfit, ECE -> 0.0)

[3] Generate predictions at a given prob_threshold
     └─ pred = (prob >= prob_thr)
     └─ error_label = (pred != true_label)   # whether the base model's prediction was wrong

[4] Train Q-model (a "prediction correctness" classifier), on the TRAIN split
     └─ Input: base features + mask + base prob + calibrated probs (Platt, Isotonic)
               + any uncertainty stats the base model produces
     └─ Output: q_prob, the predicted probability that the base prediction is wrong (error_label)
     └─ Model types: Logistic Regression / MLP / XGBoost
     └─ Training strategies: Simple (fit on full train set) vs CrossFit (5-fold cross-fitting)

[5] Filter alarms using a Q-model threshold (q_thr)
     └─ Suppress the alarm for any sample where q_prob >= q_thr
     └─ Best operating point = minimum absolute (FP+FN) within the region
        where both sensitivity and specificity are >= 0.80
     └─ If no threshold satisfies that region, fall back to the
        highest-achievable-sensitivity threshold (flagged as target_met=False)
```

In short, this is a **post-hoc filtering** approach: if the base model raises an alarm but the Q-model judges that this particular prediction is likely wrong, the alarm is suppressed.

### Supported Base Models (4, per target)

| Base Model | Features used | Uncertainty info |
|---|---|---|
| BasicMLP | Raw features + mask | None (probability only) |
| MC Dropout | Raw features + mask | variance, entropy (T=50 forward passes) |
| Deep Ensemble | Raw features + mask | variance, entropy, spread (M=5 members) |
| XGBoost | Raw features + mask | None (probability only) |

> mask: a binary indicator of missingness (based on `notna()`; 1 = value present, 0 = missing). All 4 base models operate in the same feature space (raw features + mask).

### Q-model Input Features by Base Model

For every base model, the Q-model input is **base features (+ mask) + the base model's predicted probability + calibrated versions of that probability (Platt, Isotonic)**, plus any uncertainty statistics the base model produces. Uncertainty features (variance, entropy, spread) are carried through unchanged — calibration does not apply to them, since they are not probabilities.

| Base Model | Base features + mask | + base prob | + Platt | + Isotonic | + uncertainty stats |
|---|---|---|---|---|---|
| BasicMLP | ✅ | ✅ | ✅ | ✅ | — |
| MC Dropout | ✅ | ✅ | ✅ | ✅ | variance, entropy |
| Deep Ensemble | ✅ | ✅ | ✅ | ✅ | variance, entropy, spread |
| XGBoost | ✅ | ✅ | ✅ | ✅ | — |

---

## 2. Directory Structure

```
Q-Model/
├── config.py                     # Central config: DATA_PATH, CKPT_ROOT, SRC_DIR
│                                  # (env-var overridable: QMODEL_DATA_PATH, QMODEL_CKPT_ROOT)
├── .gitignore
│
├── src/
│   └── clinical_ts/               # Shared model definitions (encoder, conv1d modules)
│
├── checkpoints/                   # Trained base-model checkpoints (.pt, gitignored)
│   ├── icu24h/
│   │   ├── basicmlp/best_basicmlp_icu24h_only.pt
│   │   ├── deepensemble/ensemble_member_{0..4}_icu24h_only_mask.pt
│   │   └── mcdropout/best_mcdropout_icu24h_only_mask.pt
│   └── mortality/
│       ├── basicmlp/best_basicmlp_mortality365d_only.pt
│       ├── deepensemble/ensemble_member_{0..4}_mortality365d_only_mask.pt
│       └── mcdropout/best_mcdropout_mortality365d_only_mask.pt
│   # (XGBoost has no checkpoint -- retrained inline every run, it's cheap)
│
├── experiments/
│   ├── icu24h/
│   │   ├── basicmlp/       {train.py, calibrate.py, train_qmodel.py}
│   │   ├── deepensemble/   {train.py, calibrate.py, train_qmodel.py}
│   │   ├── mcdropout/      {train.py, calibrate.py, train_qmodel.py}
│   │   └── xgboost/        {calibrate.py, train_qmodel.py}   # no train.py -- inline
│   └── mortality/
│       ├── basicmlp/       {train.py, calibrate.py, train_qmodel.py}
│       ├── deepensemble/   {train.py, calibrate.py, train_qmodel.py}
│       ├── mcdropout/      {train.py, calibrate.py, train_qmodel.py}
│       └── xgboost/        {train.py, calibrate.py, train_qmodel.py}
│   # each folder also gets, at run time:
│   #   results/csv/  -- calibrated_probs*.npz, calibration diagnostics, sweep summary csv
│   #   results/png/  -- training curves (train.py only)
│
├── analysis/
│   ├── shapley_icu24h.py          # SHAP + feature importance, confirmed best config per model
│   └── shapley_mortality.py       # same, for mortality365d
│
└── results/
    ├── icu24h/
    │   ├── figures/                # shap_beeswarm_*.png + top5 comparison plots (committed)
    │   └── artifacts/               # feature_importance_*.csv, shap_values_*.npy,
    │                                #   shap_importance_*.csv, qmodels/*.{joblib,pt}
    └── mortality/
        ├── figures/
        └── artifacts/
```

**Path resolution rules (see `config.py`):**
- `SRC_DIR`, checkpoint save/load locations, and each experiment's `results/csv|png` are always resolved relative to the repo / the script's own location -- portable across machines/accounts by construction.
- `DATA_PATH` (the shared clinical CSV, outside the repo) and `CKPT_ROOT` (checkpoint root) default to this account's HPC paths but can be overridden per-machine with `QMODEL_DATA_PATH` / `QMODEL_CKPT_ROOT` env vars, without touching any script.

---

## 3. How to Run

Each base model goes through up to 3 stages: **(1) train base model -> (2) calibrate + extract Q-model input features (npz) -> (3) train Q-model + sweep prob_thr**. XGBoost skips stage 1 for `icu24h` (trained inline in `calibrate.py`, since it's cheap); `mortality/xgboost/train.py` exists only to explore the threshold range, it does not persist a model.

Run from anywhere -- paths are resolved via `config.py`, not the working directory:

```bash
python experiments/<target>/<model>/train.py          # (skip for icu24h/xgboost)
python experiments/<target>/<model>/calibrate.py
python experiments/<target>/<model>/train_qmodel.py
```
where `<target>` in `{icu24h, mortality}` and `<model>` in `{basicmlp, deepensemble, mcdropout, xgboost}`.

**Example -- BasicMLP, icu24h:**
```bash
python experiments/icu24h/basicmlp/train.py
# -> checkpoints/icu24h/basicmlp/best_basicmlp_icu24h_only.pt

python experiments/icu24h/basicmlp/calibrate.py
# -> experiments/icu24h/basicmlp/results/csv/calibrated_probs.npz
# -> experiments/icu24h/basicmlp/results/csv/calibration_diagnostics_train_test.csv

python experiments/icu24h/basicmlp/train_qmodel.py
# -> experiments/icu24h/basicmlp/results/csv/prob_thr_sweep_summary_icu24h_only_mask_trainQ_WITH_CALIBRATION.csv
```

**Example -- XGBoost, mortality365d:**
```bash
python experiments/mortality/xgboost/train.py       # optional: explore PROB_THRESHOLD range only
python experiments/mortality/xgboost/calibrate.py    # trains + calibrates inline
python experiments/mortality/xgboost/train_qmodel.py
```

**Common output columns (`prob_thr_sweep_summary_*.csv`)**

| Column | Description |
|---|---|
| `prob_thr` | Decision threshold of the base model |
| `strategy` | Baseline / Simple / CrossFit |
| `model` | Q-model type (LR / MLP / XGB) |
| `base_sensitivity` | Baseline sensitivity at this prob_thr |
| `best_q_thr` | Q-model threshold that minimizes FP+FN while keeping sens & spec >= 0.80 |
| `sensitivity`, `TP`, `FP`, `FN`, `TN` | Confusion matrix at the selected operating point |
| `target_met` | Whether the sens/spec >= 0.80 region was reachable; if not, the row reports the highest-achievable-sensitivity threshold instead |

Specificity, FP/FN reduction (%), and total reduction (%) shown in Section 5 are derived post-hoc: `specificity = TN/(TN+FP)`, `FP_reduction = (FP_base - FP_new)/FP_base * 100`, `FN_reduction = (FN_base - FN_new)/FN_base * 100`, `total_reduction = ((FP_base+FN_base) - (FP_new+FN_new))/(FP_base+FN_base) * 100`.

---

## 4. Feature Importance / SHAP Analysis

```bash
python analysis/shapley_icu24h.py
python analysis/shapley_mortality.py
```

For each base model, feature importance is computed for its **confirmed best (threshold, strategy, Q-model)** combination (see Section 5 tables) -- not uniformly the same Q-model type for all four. Gain importance (XGB Q-models) and SHAP (LR/MLP Q-models, via LinearExplainer / DeepExplainer) are on different scales, so both are normalized to max = 1.0 **within each base model** before being placed on the same comparison plot -- only relative ranking/shape within a model is comparable across models, not raw magnitude.

Outputs (re-running overwrites the committed figures):
- `results/<target>/figures/shap_beeswarm_<model>.png` -- per-model SHAP beeswarm
- `results/<target>/figures/qmodel_{feature_importance,shap}_top5_4models_comparison.png` -- side-by-side top-5 across all 4 models
- `results/<target>/artifacts/` -- `feature_importance_*.csv`, `shap_values_*.npy`, `shap_importance_*.csv`
- `results/<target>/artifacts/qmodels/` -- every winning Q-model, saved so re-running never needs to retrain from scratch

**Key SHAP finding:** entropy dominates for Deep Ensemble and MC Dropout (their uncertainty-derived feature); calibration features (Platt/Isotonic) dominate for BasicMLP and XGBoost, which have no native uncertainty estimate. XGBoost also assigns exactly zero gain importance to entropy where entropy is a deterministic function of mean probability -- expected behavior, not a bug.

---

## 5. Results

### 5.1 ICU24H

**Baseline (no Q-model filtering), at the confirmed best threshold:**

| Base Model | prob_thr | Sensitivity | Specificity | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| BasicMLP | 0.10 | 0.8110 | 0.8529 | 592 | 783 | 138 | 4538 |
| Deep Ensemble | 0.10 | 0.8356 | 0.8477 | 610 | 810 | 120 | 4511 |
| MC Dropout | 0.13 | 0.8000 | 0.8566 | 584 | 763 | 146 | 4558 |
| XGBoost | 0.11 | 0.8000 | 0.8703 | 584 | 690 | 146 | 4631 |

**With Q-model filtering (best operating point per base model):**

| Base Model | prob_thr | q_thr | Best Q-model | Sensitivity | Specificity | TP | FP | FN | TN | AUROC | FP Reduction | FN Reduction | Total Reduction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BasicMLP | 0.10 | 0.86 | Simple XGB | 0.8000 | 0.8621 | 584 | 734 | 146 | 4587 | 0.9322 | 6.26% | -5.80% | 4.45% |
| Deep Ensemble | 0.10 | 0.74 | Simple MLP | 0.8014 | 0.8730 | 585 | 676 | 145 | 4645 | 0.9202 | 16.54% | -20.83% | 11.72% |
| MC Dropout | 0.13 | 0.78 | CrossFit MLP | 0.8000 | 0.8575 | 584 | 758 | 146 | 4563 | 0.9193 | 0.66% | 0.00% | 0.55% |
| XGBoost | 0.11 | 0.83 | CrossFit LR | 0.8000 | 0.8705 | 584 | 689 | 146 | 4632 | 0.8132 | 0.14% | 0.00% | 0.12% |

**Notes:**
- **Deep Ensemble shows by far the strongest Q-model improvement** (+2.53pp specificity, 16.54% FP reduction) -- also the largest FN increase (-20.83%), which is a real clinical trade-off, not just an aggregate win.
- MC Dropout and XGBoost see only marginal improvement (FP reduction < 1%) at these particular thresholds -- their baseline is already close to the sens/spec >= 0.80 boundary, leaving little "recoverable" FP.
- XGBoost's Q-model AUROC (0.8132) is notably lower than the other three (~0.92-0.93) despite using CrossFit-LR -- consistent with XGBoost lacking a native uncertainty signal for the Q-model to lean on.

### 5.2 Mortality (365-day)

**Baseline (no Q-model filtering), at the confirmed best threshold:**

| Base Model | prob_thr | Sensitivity | Specificity | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| BasicMLP | 0.11 | 0.8048 | 0.7541 | 672 | 1287 | 163 | 3947 |
| Deep Ensemble | 0.14 | 0.8144 | 0.7549 | 680 | 1283 | 155 | 3951 |
| MC Dropout | 0.15 | 0.8132 | 0.7463 | 679 | 1328 | 156 | 3906 |
| XGBoost | 0.11 | 0.8335 | 0.7381 | 696 | 1371 | 139 | 3863 |

**With Q-model filtering (best operating point per base model):**

| Base Model | prob_thr | q_thr | Best Q-model | Sensitivity | Specificity | TP | FP | FN | TN | AUROC | FP Reduction | FN Reduction | Total Reduction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BasicMLP | 0.11 | 0.84 | CrossFit XGB | 0.8024 | 0.7663 | 670 | 1223 | 165 | 4011 | 0.9297 | 4.97% | -1.23% | 4.28% |
| Deep Ensemble | 0.14 | 0.84 | Simple XGB | 0.8072 | 0.7726 | 674 | 1190 | 161 | 4044 | 0.9301 | 7.25% | -3.87% | 6.05% |
| MC Dropout | 0.15 | 0.79 | Simple MLP | 0.8060 | 0.7574 | 673 | 1270 | 162 | 3964 | 0.9181 | 4.37% | -3.85% | 3.50% |
| XGBoost | 0.11 | 0.92 | Simple XGB | 0.8048 | 0.7528 | 672 | 1294 | 163 | 3940 | 0.9155 | 5.62% | -17.27% | 3.51% |

**Notes:**
- Positive rate for 365-day mortality is ~13.8% (835/6069 in the test split above) -- moderately imbalanced, unlike the much rarer targets some earlier iterations of this project explored.
- **Deep Ensemble again shows the strongest improvement** (7.25% FP reduction, 6.05% total reduction) among the four.
- **XGBoost's FN reduction is the worst of the four (-17.27%)** despite a respectable FP reduction (5.62%) -- i.e. its Q-model buys FP reduction at a comparatively steep sensitivity cost.
- All four models show negative FN reduction (FN increases) at their best operating point -- consistent with the ICU24H results, this is an expected and clinically-relevant trade-off of post-hoc alarm suppression, not an artifact.

> Both tables' "best operating point" is defined as the threshold minimizing absolute (FP+FN) within the region where both sensitivity and specificity are >= 0.80 (Section 1, step 5). See the full `prob_thr_sweep_summary_*_WITH_CALIBRATION.csv` per model for the complete threshold-by-threshold sweep.

---

## 6. Known Limitations / Open Questions

- **Two-threshold structure**: alarms currently require passing both `prob_thr` (base model) and `q_thr` (Q-model). Whether this should collapse into a single `q_thr`-only decision (redefining the Q-model's target away from `error_label`) is an open question raised during review, not yet resolved.
- Calibrators (Platt, Isotonic) are always fit on the VAL split and evaluated on a disjoint split (never fit/eval on the same data) -- see `calibrate.py` in each model folder.
