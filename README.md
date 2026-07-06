# Q-model

A pipeline for reducing false alarms (FPs) in an ICU deterioration (24h ICU admission) prediction system, by training a separate **Q-model** that predicts whether the base model's prediction is **correct or incorrect**.

## 1. Overview

Clinical early-warning models are typically operated at a fixed sensitivity (recall) threshold (≥ 0.80). At this operating point, the number of false alarms (FPs) tends to be high, leading to alarm fatigue.

The Q-model addresses this problem as follows:

```
[1] Train Base Model
     └─ Input: structured patient data (demographics, vitals, labvalues, biometrics)
     └─ Output: predicted probability of icu_24h (prob)

[2] Generate predictions at a given prob_threshold
     └─ pred = (prob >= prob_thr)
     └─ error_label = (pred != true_label)   # whether the base model's prediction was wrong

[3] Train Q-model (a "prediction correctness" classifier)
     └─ Input: base features + base model outputs (prob, uncertainty, etc.)
     └─ Output: q_prob, the predicted probability that the base prediction is wrong (error_label)
     └─ Model types: Logistic Regression / MLP / XGBoost
     └─ Training strategies: Simple (fit on full train set) vs CrossFit (5-fold cross-fitting)

[4] Filter alarms using a Q-model threshold (q_thr)
     └─ Suppress the alarm for any sample where q_prob >= q_thr
     └─ Maximize the FP reduction while keeping sensitivity >= 0.80
```

In short, this is a **post-hoc filtering** approach: if the base model raises an alarm but the Q-model judges that this particular prediction is likely wrong, the alarm is suppressed.

### Supported Base Models (4)

| Base Model | Features used | Uncertainty info |
|---|---|---|
| BasicMLP | Raw features + mask | None (probability only) |
| MC Dropout | Raw features + mask | variance, entropy (T=50 forward passes) |
| Deep Ensemble | Raw features + mask | variance, entropy, spread (M=5 members) |
| XGBoost | Raw features + mask | None (probability only) |

> mask: a binary indicator of missingness (based on `notna()`; 1 = value present, 0 = missing). All 4 base models now include mask columns as features, so they operate in the same feature space (raw features + mask).

### Q-model Input Features by Base Model

For every base model, the Q-model input is **base features (+ mask) + the base model's predicted probability**, plus any uncertainty statistics the base model produces:

| Base Model | Base features + mask | + base prob | + uncertainty stats |
|---|---|---|---|
| BasicMLP | ✅ | ✅ | — |
| MC Dropout | ✅ | ✅ | variance, entropy |
| Deep Ensemble | ✅ | ✅ | variance, entropy, spread |
| XGBoost | ✅ | ✅ | — |

---

## 2. Directory Structure

```
Q-MODEL/
├── basicmlp/
│   ├── basicMLP.py                                   # Trains the base model + extracts Q-model input features
│   ├── best_basicmlp_icu24h_only.pt                  # Trained base model checkpoint
│   ├── qmodel_basicMLP.py                            # Trains Q-model + runs prob_thr sweep
│   └── prob_thr_sweep_summary_icu24h_only_mask_trainQ.csv   # Results summary
│
├── deepensemble/
│   ├── ensemble.py                                   # Trains base model (M=5 members) + extracts features
│   ├── ensemble_member_{0..4}_icu24h_only_mask.pt     # Ensemble member checkpoints
│   ├── qmodel_deepensemble.py                         # Trains Q-model + runs prob_thr sweep
│   └── prob_thr_sweep_summary_ensemble_icu24h_only_mask_trainQ.csv
│
├── mcdropout/
│   ├── MC.py                                          # Trains base model + runs MC Dropout inference
│   ├── best_mcdropout_icu24h_only_mask.pt
│   ├── qmodel_MC.py                                   # Trains Q-model + runs prob_thr sweep
│   └── prob_thr_sweep_summary_mcdropout_icu24h_only_mask_trainQ.csv
│
├── xgboost/
│   ├── qmodel_xgboost.py                              # Trains base model (with mask) + Q-model + sweep (all in one script)
│   └── prob_thr_sweep_summary_xgboost_withmask_trainQ.csv
│
└── feature/
    ├── feature.py                                     # Computes & compares feature importance across all 4 models
    ├── feature_importance_{basicmlp,deepensemble,mcdropout,xgboost}.csv
    ├── qmodel_feature_importance_top5_4models_comparison.png
    └── fig/                                            # Per-model feature importance plots
```

> [!NOTE]
> To confirm: the file roles above are inferred from filenames and prior discussion. Please double-check details such as whether `qmodel_xgboost.py` really includes base model training, and correct anything that doesn't match the actual code.

---

## 3. How to Run

Each base model is run in two stages: **(1) train the base model → (2) train the Q-model and run the threshold sweep**. (For XGBoost, both stages are combined into a single script.)

### 3.1 BasicMLP

```bash
# 1) Train base model (single target: icu_24h) + extract Q-model input features
python basicmlp/basicMLP.py
# → best_basicmlp_icu24h_only.pt
# → results/csv/q_features_{val,test}_basicmlp_icu24h_only.csv

# 2) Train Q-model (LR/MLP/XGB × Simple/CrossFit) + sweep prob_thr (0.05–0.20)
python basicmlp/qmodel_basicMLP.py
# → prob_thr_sweep_summary_icu24h_only_mask_trainQ.csv
```

### 3.2 Deep Ensemble

```bash
# 1) Train ensemble (M=5 members)
python deepensemble/ensemble.py
# → ensemble_member_{0..4}_icu24h_only_mask.pt

# 2) Train Q-model + sweep (uncertainty features: variance, entropy, spread)
python deepensemble/qmodel_deepensemble.py
# → prob_thr_sweep_summary_ensemble_icu24h_only_mask_trainQ.csv
```

### 3.3 MC Dropout

```bash
# 1) Train base model
python mcdropout/MC.py
# → best_mcdropout_icu24h_only_mask.pt

# 2) Train Q-model + sweep (uncertainty features: variance, entropy, T=50)
python mcdropout/qmodel_MC.py
# → prob_thr_sweep_summary_mcdropout_icu24h_only_mask_trainQ.csv
```

### 3.4 XGBoost

```bash
# Base model training (now includes mask features, same space as the other 3 models)
# + Q-model training + sweep, all combined in one script
python xgboost/qmodel_xgboost.py
# → prob_thr_sweep_summary_xgboost_withmask_trainQ.csv
```

**Common output columns (`prob_thr_sweep_summary_*.csv`)**

| Column | Description |
|---|---|
| `prob_thr` | Decision threshold of the base model (0.05–0.20) |
| `strategy` | Baseline / Simple / CrossFit |
| `model` | Q-model type (LR / MLP / XGB) |
| `base_sensitivity` | Baseline sensitivity at this prob_thr |
| `best_q_thr` | Q-model threshold that maximizes FP reduction while keeping sens ≥ 0.80 |
| `sensitivity`, `TP`, `FP`, `FN`, `TN` | Confusion matrix at the selected operating point |
| `FP_reduction_pct` | FP reduction (%) relative to baseline |

---

## 4. Computing Feature Importance

```bash
python feature/feature.py
```

- Computes feature importance for the (XGB-based) Q-model trained on each of the 4 base models (BasicMLP, Deep Ensemble, MC Dropout, XGBoost).
- Per-model results: `feature_importance_{model_name}.csv`
- A side-by-side comparison of the top-5 features across all 4 models: `qmodel_feature_importance_top5_4models_comparison.png`
- Per-model detailed plots are saved under `fig/`.

> [!NOTE]
> To confirm: does feature importance use only the XGB Q-model, or is it computed for both the Simple and CrossFit strategies? Let me know and I'll update this section accordingly.

**Top-5 feature importance comparison across all 4 models:**

![Feature importance comparison](feature/qmodel_feature_importance_top5_4models_comparison.png)

---

## 5. Results

### 5.1 Baseline Model Performance

| Model | Threshold | Sensitivity | Specificity | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|
| BasicMLP | 0.08 | 0.8466 | 0.8199 | 618 | 958 | 4363 | 112 |
| MC Dropout | 0.10 | 0.8438 | 0.8134 | 616 | 993 | 4328 | 114 |
| Deep Ensemble | 0.08 | 0.8575 | 0.8109 | 626 | 1006 | 4315 | 104 |
| XGBoost | 0.09 | 0.8000 | 0.8414 | 584 | 844 | 4477 | 146 |

### 5.2 Performance with Q-model Filtering (Best Operating Point per Base Model)

| Base Model | Threshold | Strategy | Q-model | Sensitivity | Specificity | FP Reduction | FN Reduction | TP | FP | TN | FN | Total Reduction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BasicMLP | 0.09 | Simple | XGB | 0.8041 | 0.8635 | 14.37% | -13.49% | 587 | 727 | 4594 | 143 | 10.77% |
| Deep Ensemble | 0.10 | Simple | XGB | 0.8096 | 0.8698 | 14.44% | -15.83% | 591 | 693 | 4628 | 139 | 10.54% |
| MC Dropout | 0.11 | CrossFit | XGB | 0.8055 | 0.8571 | 17.19% | -14.52% | 588 | 761 | 4560 | 142 | 13.42% |
| XGBoost | 0.09 | Simple | XGB | 0.8000 | 0.8414 | 0.00% | 0.00% | 584 | 844 | 4477 | 146 | 0.00% |

**Conclusions**
- Across BasicMLP, Deep Ensemble, and MC Dropout, the **XGB-based Q-model** yields a positive total reduction (FP+FN) over the baseline, ranked **MC Dropout (13.42%) > BasicMLP (10.77%) > Deep Ensemble (10.54%)**.
- The **Simple** strategy is best for BasicMLP and Deep Ensemble, while **CrossFit** is best for MC Dropout.
- For all three, **FN increases** relative to baseline (negative FN reduction) while **FP decreases substantially** — the Q-model trades a small increase in missed cases for a much larger drop in false alarms, and the net effect is still a reduction in total errors.
- **XGBoost shows 0% reduction** at its own best operating point — the Q-model was not able to improve on the XGBoost baseline at threshold 0.09.

> [!NOTE]
> The XGBoost row above still reflects the mask-included re-run's baseline threshold (0.09) with no Q-model improvement. Please confirm this is expected, or let me know if a different threshold/strategy should be used once new sweep results are available.
