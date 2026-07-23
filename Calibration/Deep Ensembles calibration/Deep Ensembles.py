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
import os
DATA_PATH=r"C:\Users\Taki Djebbar\Documents\Data Science S1\Medical Data Analysis with Deep Learning\Other\mds_ed.csv"
BASE_DIR    = r"C:\user\gaad2403\MDS-ED\key\Final\DeepEnsemble"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
PNG_DIR     = os.path.join(RESULTS_DIR, "png")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

PROB_THRESHOLDS = np.round(np.arange(0.05, 0.21, 0.01), 2)
Q_THRESHOLDS    = np.round(np.arange(0.00, 1.01, 0.01), 2)
BATCH_SIZE      = 32
LIN_FTRS        = [128, 128, 128]
M               = 5
EPSILON         = 1e-10
TARGET_TASK     = "mortality_365d"     # <-- must match training script
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"prob_threshold sweep: {PROB_THRESHOLDS}")
print(f"Device: {DEVICE}")

# ============================================================
# Model definitions (unchanged)
# ============================================================
class BasicEncoderStatic(EncoderStaticBase):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        self.input_channels_cat  = hparams_input_shape.static_dim_cat
        self.input_channels_cont = hparams_input_shape.static_dim
        assert(len(hparams_encoder_static.embedding_dims) == hparams_input_shape.static_dim_cat
               and len(hparams_encoder_static.vocab_sizes) == hparams_input_shape.static_dim_cat)
        self.embeddings = nn.ModuleList()
        for v, e in zip(hparams_encoder_static.vocab_sizes, hparams_encoder_static.embedding_dims):
            self.embeddings.append(nn.Embedding(v, e))
        self.input_dim = int(np.sum(hparams_encoder_static.embedding_dims) + hparams_input_shape.static_dim)

    def embed(self, **kwargs):
        static     = kwargs.get("static", None)
        static_cat = kwargs.get("static_cat", None)
        res = []
        if static_cat is not None:
            for i, e in enumerate(self.embeddings):
                res.append(e(static_cat[:, i].long()))
            res = torch.cat([torch.cat(res, dim=1), static], dim=1) if static is not None else torch.cat(res, dim=1)
        else:
            res = static
        return res

    def forward(self, **kwargs): raise NotImplementedError
    def get_output_shape(self):  raise NotImplementedError


class BasicEncoderStaticMLP(BasicEncoderStatic):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        lin_ftrs = [self.input_dim] + list(hparams_encoder_static.lin_ftrs)
        if target_dim is not None and lin_ftrs[-1] != target_dim:
            lin_ftrs.append(target_dim)
        ps = ([hparams_encoder_static.dropout]
              if not isinstance(hparams_encoder_static.dropout, Iterable)
              else hparams_encoder_static.dropout)
        if len(ps) == 1:
            ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
        actns  = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
        layers = []
        for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
            layers += bn_drop_lin(ni, no, hparams_encoder_static.batch_norm, p, actn, layer_norm=False)
        self.layers = nn.Sequential(*layers)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.static_dim     = int(lin_ftrs[-1])
        self.output_shape.static_dim_cat = 0

    def forward(self, **kwargs):
        return {"static": self.layers(self.embed(**kwargs))}

    def get_output_shape(self):
        return self.output_shape


@dataclass
class MLPConfig:
    embedding_dims: List[int] = field(default_factory=list)
    vocab_sizes: List[int]    = field(default_factory=list)
    lin_ftrs: List[int]       = field(default_factory=lambda: [128, 128, 128])
    dropout: float   = 0.5
    batch_norm: bool = True

@dataclass
class ShapeCfg:
    static_dim: int     = 0
    static_dim_cat: int = 0
    channels: int       = 0
    length: int         = 0
    sequence_last: bool = False
    channels2: int      = 0

# ============================================================
# 1. Load & preprocess data — IDENTICAL to training script
# ============================================================
print("\nLoading data...")
df = pd.read_csv(DATA_PATH
                 , low_memory=False)

input_cols = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]

# --- missingness masks (this is what training used and this script was missing) ---
mask_columns = []
for c in input_cols:
    mask_col = c + '_m'
    df[mask_col] = df[c].notna().astype(float)
    mask_columns.append(mask_col)

df_train      = df[df['general_strat_fold'] < 18]
train_medians = df_train[input_cols].median().to_dict()
train_nans    = [c for c, v in df_train[input_cols].isna().sum().to_dict().items() if v > 0]
for c in train_nans:
    df.loc[df[c].isna(), c] = train_medians[c]
df = df.copy()

unique_counts = {c: len(np.unique(np.array(df[c]))) for c in input_cols}
cat_features  = [c for c, v in unique_counts.items()
                 if v < 10 and not c.endswith("nan") and not c.startswith("labvalues")]
cont_features = [c for c in input_cols if c not in cat_features]
cont_features = cont_features + mask_columns   # <-- masks appended, same as training

df["vitals_acuity"] = df["vitals_acuity"].apply(lambda x: int(x) - 1)

lbl_eth = ['demographics_ethnicity_asian','demographics_ethnicity_black/african',
           'demographics_ethnicity_hispanic/latino','demographics_ethnicity_other',
           'demographics_ethnicity_white']
df["demographics_ethnicity"] = df.apply(lambda r: np.where([r[c] for c in lbl_eth])[0][0], axis=1)
df.drop(lbl_eth, axis=1, inplace=True)

# --- drop the ethnicity mask columns, same as training ---
ethnicity_masks = [c + '_m' for c in lbl_eth if (c + '_m') in df.columns]
if ethnicity_masks:
    df.drop(ethnicity_masks, axis=1, inplace=True)
    mask_columns = [c for c in mask_columns if c not in ethnicity_masks]

input_cols    = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]
cat_features  = [c for c in input_cols if c in cat_features]
cont_features = [c for c in input_cols if c not in cat_features]

df["deterioration_" + TARGET_TASK] = df["deterioration_" + TARGET_TASK].replace(-999., np.nan)

val_df  = df[df['general_strat_fold'] == 18].reset_index(drop=True)
test_df = df[df['general_strat_fold'] == 19].reset_index(drop=True)
val_df  = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
print(f"Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================
# 2. Dataset & DataLoader — single target task, matches training
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, df, cont_features, cat_features, target_task):
        self.cont   = torch.tensor(df[cont_features].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_features].values,  dtype=torch.long)
        self.labels = torch.tensor(df[["deterioration_" + target_task]].values, dtype=torch.float32)
    def __len__(self): return len(self.cont)
    def __getitem__(self, idx): return self.cont[idx], self.cat[idx], self.labels[idx]

val_loader  = DataLoader(TabularDataset(val_df,  cont_features, cat_features, TARGET_TASK),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(TabularDataset(test_df, cont_features, cat_features, TARGET_TASK),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================
# 3. Load ensemble members & inference
# ============================================================
shape   = ShapeCfg(static_dim=len(cont_features), static_dim_cat=len(cat_features))
mlp_cfg = MLPConfig(
    embedding_dims=[unique_counts[c] for c in cat_features],
    vocab_sizes=[unique_counts[c] for c in cat_features],
    lin_ftrs=LIN_FTRS
)

print(f"\nLoading {M} ensemble members...")
ensemble_models = []
for m in range(M):
    pt_path = os.path.join(BASE_DIR, f"ensemble_member_{m}_mortality365d_only_mask.pt")
    model   = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=1).to(DEVICE)   # <-- target_dim=1, matches training
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=False))
    model.eval()
    ensemble_models.append(model)
    print(f"  Loaded: {pt_path}")


def ensemble_predict(models, loader):
    all_preds_per_model = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for cont, cat, _ in loader:
                cont, cat = cont.to(DEVICE), cat.to(DEVICE)
                probs = torch.sigmoid(model(static=cont, static_cat=cat)["static"]).cpu().numpy()
                preds.append(probs)
        all_preds_per_model.append(np.concatenate(preds, axis=0))

    all_labels = []
    for _, _, labels in loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels, axis=0)

    stacked   = np.stack(all_preds_per_model, axis=0)
    mean_pred = stacked.mean(axis=0)   # shape (N, 1)
    variance  = stacked.var(axis=0)
    spread    = stacked.max(axis=0) - stacked.min(axis=0)
    p         = mean_pred
    entropy   = -(p * np.log(p + EPSILON) + (1-p) * np.log(1-p + EPSILON))

    return mean_pred, variance, entropy, spread, all_labels


print("Running ensemble inference...")
print("  Val set...")
val_mean,  val_var,  val_ent,  val_spr,  val_labels  = ensemble_predict(ensemble_models, val_loader)
print("  Test set...")
test_mean, test_var, test_ent, test_spr, test_labels = ensemble_predict(ensemble_models, test_loader)
print("  Done.")

# mean_pred has shape (N, 1) since the model is single-task (mortality only)
test_prob_mortality365d = test_mean[:, 0]
print(f"\nTest set probabilities (mortality 365d) per sample:")
print(f"  Shape : {test_prob_mortality365d.shape}")
print(f"  Min   : {test_prob_mortality365d.min():.4f}")
print(f"  Max   : {test_prob_mortality365d.max():.4f}")
print(f"  Mean  : {test_prob_mortality365d.mean():.4f}")

###############################################
#######  PLOT #############

# ============================================================
# Calibration Analysis for both classes
# ============================================================
print("\n" + "="*70)
print("Calibration Analysis (Positive and Negative classes)")
print("="*70)

# Apply mask to get only valid (non-NaN) samples
test_mask     = ~np.isnan(test_labels[:, ICU24H_IDX])
test_prob_icu_deep_ensembles = test_mean[test_mask, ICU24H_IDX]
test_true_icu_deep_ensembles = test_labels[test_mask, ICU24H_IDX].astype(int)

print(f"Total test samples : {len(test_true_icu_deep_ensembles)}")
print(f"Positive (label=1) : {test_true_icu_deep_ensembles.sum()}")
print(f"Negative (label=0) : {(test_true_icu_deep_ensembles == 0).sum()}")

# ECE function
def compute_ece(true_labels, pred_probs, n_bins=30):
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    total = len(pred_probs)
    for i in range(n_bins):
        mask      = (pred_probs >= bins[i]) & (pred_probs < bins[i+1])
        bin_count = mask.sum()
        if bin_count == 0:
            continue
        ece += (bin_count / total) * abs(true_labels[mask].mean() - pred_probs[mask].mean())
    return ece

# ============================================================
# Compute bins manually for both classes
# ============================================================
n_bins    = 30
bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
bin_width = bin_edges[1] - bin_edges[0]
centers   = (bin_edges[:-1] + bin_edges[1:]) / 2

# --- Positive class ---
frac_pos_DE = np.full(n_bins, np.nan)
conf_pos     = np.full(n_bins, np.nan)
for i in range(n_bins):
    mask = (test_prob_icu_deep_ensembles >= bin_edges[i]) & (test_prob_icu_deep_ensembles < bin_edges[i+1])
    if mask.sum() > 0:
        frac_pos_DE[i] = test_true_icu_deep_ensembles[mask].mean()
        conf_pos[i]     = test_prob_icu_deep_ensembles[mask].mean()

# --- Negative class ---
neg_probs    = 1 - test_prob_icu_deep_ensembles
neg_true     = 1 - test_true_icu_deep_ensembles
frac_neg_DE = np.full(n_bins, np.nan)
conf_neg     = np.full(n_bins, np.nan)
for i in range(n_bins):
    mask = (neg_probs >= bin_edges[i]) & (neg_probs < bin_edges[i+1])
    if mask.sum() > 0:
        frac_neg_DE[i] = neg_true[mask].mean()
        conf_neg[i]     = neg_probs[mask].mean()

ece_pos_DE = compute_ece(test_true_icu_deep_ensembles, test_prob_icu_deep_ensembles,   n_bins=30)
ece_neg_DE = compute_ece(neg_true,      neg_probs,        n_bins=30)

brier_pos = brier_score_loss(test_true_icu_deep_ensembles, test_prob_icu_deep_ensembles)
brier_neg = brier_score_loss(neg_true,      neg_probs)

print(f"\nExpected Calibration Error (ECE):")
print(f"  Positive class (label=1) ECE : {ece_pos_DE:.4f}")
print(f"  Negative class (label=0) ECE : {ece_neg_DE:.4f}")
print(f"\nBrier Score:")
print(f"  Positive class (label=1) : {brier_pos:.4f}")
print(f"  Negative class (label=0) : {brier_neg:.4f}")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# ------------------------------------------------------------
# Positive class
# ------------------------------------------------------------
axes[0].plot([0, 1], [0, 1], '--', color='gray', linewidth=1,
             label='Perfect Calibration')
axes[0].plot(centers, frac_pos_DE, color='red', linewidth=2,
             label='Observed')
axes[0].plot(centers, conf_pos, color='black', linestyle='--',
             label='Mean predicted prob')
axes[0].bar(centers, frac_pos_DE, width=bin_width * 0.9,
            alpha=0.2, color='red')
axes[0].text(0.05, 0.90, f"ECE (positive class): {ece_pos_DE:.3f}",
             transform=axes[0].transAxes, fontsize=8)
axes[0].set_xlabel('Predicted probability of mortality_1y', fontsize=8)
axes[0].set_ylabel('Fraction of true ICU cases', fontsize=8)
axes[0].set_title('Calibration for the Positive Class (mortality_1y = 1)', fontsize=8)
axes[0].legend(loc='upper right', fontsize=6)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])
axes[0].grid(alpha=0.3)

# ------------------------------------------------------------
# Negative class
# ------------------------------------------------------------
axes[1].plot([0, 1], [0, 1], '--', color='gray', linewidth=1,
             label='Perfect Calibration')
axes[1].plot(centers, frac_neg_DE, color='blue', linewidth=2,
             label='Observed')
axes[1].plot(centers, conf_neg, color='black', linestyle='--',
             label='Mean predicted prob')
axes[1].bar(centers, frac_neg_DE, width=bin_width * 0.9,
            alpha=0.2, color='blue')
axes[1].text(0.05, 0.90, f"ECE (negative class): {ece_neg_DE:.3f}",
             transform=axes[1].transAxes, fontsize=8)
axes[1].set_xlabel('Predicted probability of non-mortality_1y', fontsize=8)
axes[1].set_ylabel('Fraction of true mortality_1y cases', fontsize=8)
axes[1].set_title('Calibration for the Negative Class (mortality_1y = 0)', fontsize=8)
axes[1].legend(loc='upper right', fontsize=6)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1])
axes[1].grid(alpha=0.3)

plt.suptitle('Deep Ensembles Calibration Analysis (mortality_1y)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PNG_DIR, "calibration_both_classes.png"),
            dpi=150, bbox_inches='tight')
plt.show()