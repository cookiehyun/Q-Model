import sys
sys.path.insert(0, "/fs/dss/home/gaad2403/MDS-ED/src")

import torch
from torch import nn
import numpy as np
import pandas as pd
import dataclasses
from dataclasses import dataclass, field
from typing import List
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from collections.abc import Iterable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from clinical_ts.template_modules import EncoderStaticBase, EncoderStaticBaseConfig
from clinical_ts.ts.basic_conv1d_modules.basic_conv1d import bn_drop_lin
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
%matplotlib inline
import io
from PIL import Image
from IPython.display import display
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
)
# ============================================================
# Paths
# ============================================================
BASE_DIR    = "/user/gaad2403/MDS-ED/key/Final/XGboost"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
PNG_DIR     = os.path.join(RESULTS_DIR, "png")
DATA_PATH   = r'C:\Users\Taki Djebbar\Documents\Data Science S1\Medical Data Analysis with Deep Learning\Other\mds_ed.csv'

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

PROB_THRESHOLDS = np.round(np.arange(0.05, 0.21, 0.01), 2)
Q_THRESHOLDS    = np.round(np.arange(0.00, 1.01, 0.01), 2)
N_FOLDS         = 5
RANDOM_STATE    = 42
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MLP_HIDDEN     = [64, 32]
MLP_EPOCHS     = 50
MLP_LR         = 1e-3
MLP_BATCH_SIZE = 64
MLP_DROPOUT    = 0.3

XGB_BASE_PARAMS = dict(random_state=RANDOM_STATE, n_jobs=4, eval_metric='logloss')

print(f"prob_threshold sweep: {PROB_THRESHOLDS}")
print(f"Device: {DEVICE}")

# ============================================================
# 1. Load & preprocess data (same as paper)
# ============================================================
print("\nLoading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"shape: {df.shape}")

demographics_columns = [c for c in df.columns if 'demographics_' in c]
biometrics_columns   = [c for c in df.columns if 'biometrics_' in c]
vitals_columns       = [c for c in df.columns if 'vitals_' in c]
labvalues_columns    = [c for c in df.columns if 'labvalues_' in c]
all_features         = demographics_columns + biometrics_columns + vitals_columns + labvalues_columns

selected_folds = df[df['general_strat_fold'].isin(range(0, 18))]
medians        = selected_folds[all_features].median()

mask_columns = []
for col in all_features:
    mask_col = col + '_m'
    df[mask_col] = df[col].isna().astype(float)
    mask_columns.append(mask_col)

df[all_features]       = df[all_features].fillna(medians)
all_features_with_mask = all_features + mask_columns
print(f"Features: {len(all_features)} + {len(mask_columns)} masks = {len(all_features_with_mask)}")

target_columns = [
    'deterioration_mortality_365d',
    'deterioration_icu_24h',
    'deterioration_cardiac_arrest',
    'deterioration_vasopressors'
]
# Choose the label ICU24H_IDX = 0
ICU24H_IDX = 0
# ============================================================
# 2. Train/Val/Test split
# ============================================================
train_df = df[df['general_strat_fold'].isin(range(0, 18))].reset_index(drop=True)
val_df   = df[df['general_strat_fold'] == 18].reset_index(drop=True)
test_df  = df[df['general_strat_fold'] == 19].reset_index(drop=True)

val_df  = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)

x_train = train_df[all_features_with_mask].values
x_val   = val_df[all_features_with_mask].values
x_test  = test_df[all_features_with_mask].values

y_train = train_df[target_columns].values
y_val   = val_df[target_columns].values
y_test  = test_df[target_columns].values

print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

# ============================================================
# 3. Train XGBoost base model (ICU 24h target only)
# ============================================================
print("\nTraining XGBoost base model (ICU 24h)...")

i = ICU24H_IDX
y_tr_raw = y_train[:, i]; y_v_raw = y_val[:, i]; y_te_raw = y_test[:, i]

mask_tr = y_tr_raw != -999
mask_v  = y_v_raw  != -999
mask_te = y_te_raw != -999

y_tr = y_tr_raw[mask_tr].astype(int)
y_v  = y_v_raw[mask_v].astype(int)
y_te = y_te_raw[mask_te].astype(int)

x_tr = x_train[mask_tr]
x_v  = x_val[mask_v]
x_te = x_test[mask_te]

val_valid_idx  = np.where(mask_v)[0]
test_valid_idx = np.where(mask_te)[0]

base_model = XGBClassifier(**XGB_BASE_PARAMS)
base_model.fit(x_tr, y_tr, eval_set=[(x_v, y_v)], verbose=False)

val_prob_icu_xgb  = base_model.predict_proba(x_v)[:, 0]
test_prob_icu_xgb = base_model.predict_proba(x_te)[:, 0]
val_true_icu_xgb  = y_v
test_true_icu_xbg = y_te

auroc_val  = roc_auc_score(y_v,  val_prob_icu_xgb)
auroc_test = roc_auc_score(y_te, test_prob_icu_xgb)
print(f"  Val  AUROC: {auroc_val:.4f}")
print(f"  Test AUROC: {auroc_test:.4f}")
print(f"  Val  samples: {len(val_prob_icu_xgb)}")
print(f"  Test samples: {len(test_prob_icu_xgb)}")


##############################################
#Plot
##############################################

# ============================================================
# Calibration Analysis for XGBoost - both classes
# ============================================================
print("\n" + "="*70)
print("XGBoost Calibration Analysis (Positive and Negative classes)")
print("="*70)

# XGBoost data already computed from the main code
xgb_prob = test_prob_icu_xgb.copy()
xgb_true = test_true_icu_xbg.copy()

# Remove NaNs
mask_valid   = ~np.isnan(xgb_prob) & ~np.isnan(xgb_true.astype(float))
xgb_prob     = xgb_prob[mask_valid]
xgb_true     = xgb_true[mask_valid].astype(int)

print(f"Total test samples : {len(xgb_true)}")
print(f"Positive (label=1) : {xgb_true.sum()}")
print(f"Negative (label=0) : {(xgb_true == 0).sum()}")

# ============================================================
# ECE function
# ============================================================
def compute_ece_by_class(y_true, y_prob, target_class=1, n_bins=30):
    y_prob = y_prob.copy()
    y_true = y_true.copy()

    if target_class == 0:
        y_prob = 1 - y_prob
        y_true = (y_true == 0).astype(int)
    else:
        y_true = (y_true == 1).astype(int)

    bins         = np.linspace(0, 1, n_bins + 1)
    bin_frac_pos = []
    bin_conf     = []
    bin_sizes    = []
    ece          = 0.0
    n            = len(y_true)

    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        bin_mask    = (y_prob >= left) & (y_prob <= right) if i == n_bins - 1 \
                      else (y_prob >= left) & (y_prob < right)
        size        = np.sum(bin_mask)

        if size == 0:
            bin_frac_pos.append(np.nan)
            bin_conf.append(np.nan)
            bin_sizes.append(0)
            continue

        frac_pos = np.sum(y_true[bin_mask] == 1) / size
        conf     = np.mean(y_prob[bin_mask])
        ece     += (size / n) * abs(frac_pos - conf)

        bin_frac_pos.append(frac_pos)
        bin_conf.append(conf)
        bin_sizes.append(size)

    return ece, bins, np.array(bin_frac_pos), np.array(bin_conf), np.array(bin_sizes)

# ============================================================
# Compute bins for both classes
# ============================================================
n_bins    = 30
bin_width = 1.0 / n_bins

xgb_ece_pos, xgb_bins_1, xgb_frac_1, xgb_conf_1, xgb_sizes_1 = compute_ece_by_class(
    xgb_true, xgb_prob, target_class=1, n_bins=n_bins)

xgb_ece_neg, xgb_bins_0, xgb_frac_0, xgb_conf_0, xgb_sizes_0 = compute_ece_by_class(
    xgb_true, xgb_prob, target_class=0, n_bins=n_bins)

xgb_centers_1 = (xgb_bins_1[:-1] + xgb_bins_1[1:]) / 2
xgb_centers_0 = (xgb_bins_0[:-1] + xgb_bins_0[1:]) / 2

xgb_brier_pos = brier_score_loss(xgb_true,       xgb_prob)
xgb_brier_neg = brier_score_loss(1 - xgb_true,   1 - xgb_prob)

print(f"\nExpected Calibration Error (ECE):")
print(f"  Positive class (label=1) ECE : {xgb_ece_pos:.4f}")
print(f"  Negative class (label=0) ECE : {xgb_ece_neg:.4f}")
print(f"\nBrier Score:")
print(f"  Positive class (label=1) : {xgb_brier_pos:.4f}")
print(f"  Negative class (label=0) : {xgb_brier_neg:.4f}")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# ------------------------------------------------------------
# Positive class (label=1)
# ------------------------------------------------------------
axes[0].plot([0, 1], [0, 1], '--', color='gray', linewidth=1,
             label='Perfect Calibration')
axes[0].plot(xgb_centers_1, xgb_frac_1, color='red', linewidth=2,
             label='Observed')
axes[0].plot(xgb_centers_1, xgb_conf_1, color='black', linestyle='--',
             label='Mean predicted prob')
axes[0].bar(xgb_centers_1, xgb_frac_1, width=bin_width * 0.9,
            alpha=0.2, color='red')
axes[0].text(0.05, 0.90, f"ECE (positive class): {xgb_ece_pos:.3f}",
             transform=axes[0].transAxes, fontsize=8)
axes[0].set_xlabel('Predicted probability of mortality', fontsize=8)
axes[0].set_ylabel('Fraction of true mortality cases', fontsize=8)
axes[0].set_title('XGBoost Calibration — Positive Class (Mortality_y1 = 1)', fontsize=8)
axes[0].legend(fontsize=8)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])
axes[0].grid(alpha=0.3)
axes[0].legend(loc='upper right', fontsize=6)
# ------------------------------------------------------------
# Negative class (label=0)
# ------------------------------------------------------------
axes[1].plot([0, 1], [0, 1], '--', color='gray', linewidth=1,
             label='Perfect Calibration')
axes[1].plot(xgb_centers_0, xgb_frac_0, color='blue', linewidth=2,
             label='Observed')
axes[1].plot(xgb_centers_0, xgb_conf_0, color='black', linestyle='--',
             label='Mean predicted prob')
axes[1].bar(xgb_centers_0, xgb_frac_0, width=bin_width * 0.9,
            alpha=0.2, color='blue')
axes[1].text(0.05, 0.90, f"ECE (negative class): {xgb_ece_neg:.3f}",
             transform=axes[1].transAxes, fontsize=8)
axes[1].set_xlabel('Predicted probability of non-mortality', fontsize=8)
axes[1].set_ylabel('Fraction of true non-mortality cases', fontsize=8)
axes[1].set_title('XGBoost Calibration — Negative Class (mortality_y1 = 0)', fontsize=8)
axes[1].legend(fontsize=8)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])
axes[1].grid(alpha=0.3)
axes[1].legend(loc='upper right', fontsize=6)
plt.suptitle('XGBoost Calibration Analysis (mortality y1)',
             fontsize=11, fontweight='bold')
plt.tight_layout()

# Force inline display
buf = io.BytesIO()
plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
buf.seek(0)
display(Image.open(buf))
plt.savefig(os.path.join(PNG_DIR, "calibration_xgboost.png"),
            dpi=150, bbox_inches='tight')
plt.close()