"""
TP vs FP uncertainty distribution plots for MC Dropout and Deep Ensemble (mortality_1d).
Loads the already-trained checkpoints (no retraining) and plots on the TEST split,
at each model's confirmed threshold:
    MC Dropout    : threshold=0.010, uncertainty = entropy (binary entropy of mean prob)
    Deep Ensemble : threshold=0.018, uncertainty = entropy (binary entropy of mean prob)
"""

import sys
sys.path.insert(0, "/fs/dss/home/gaad2403/MDS-ED/src")

import torch
from torch import nn
import numpy as np
import pandas as pd
import dataclasses
from dataclasses import dataclass, field
from typing import List
from torch.utils.data import Dataset, DataLoader
from collections.abc import Iterable
from clinical_ts.template_modules import EncoderStaticBase, EncoderStaticBaseConfig
from clinical_ts.ts.basic_conv1d_modules.basic_conv1d import bn_drop_lin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "/user/gaad2403/MDS-ED/src/data/memmap/mds_ed.csv"
OUT_DIR   = "/user/gaad2403/MDS-ED/key/Final/mortality/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

MC_PT_PATH  = "/user/gaad2403/MDS-ED/key/Final/mortality/best_mcdropout_mortality1d_only_mask.pt"
DE_PT_DIR   = "/user/gaad2403/MDS-ED/key/Final/mortality"
DE_PT_FILES = [os.path.join(DE_PT_DIR, f"ensemble_member_{m}_mortality1d_only_mask.pt") for m in range(5)]

MC_THR      = 0.010
DE_THR      = 0.018
MC_SAMPLES  = 50
EPSILON     = 1e-10
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------
# Model definitions (same as training scripts)
# ------------------------------------------------------------
class BasicEncoderStatic(EncoderStaticBase):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        self.input_channels_cat  = hparams_input_shape.static_dim_cat
        self.input_channels_cont = hparams_input_shape.static_dim
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

class TabularDataset(Dataset):
    def __init__(self, df, cont_f, cat_f, target_col):
        self.cont   = torch.tensor(df[cont_f].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_f].values,  dtype=torch.long)
        self.labels = torch.tensor(df[[target_col]].values, dtype=torch.float32)
    def __len__(self): return len(self.cont)
    def __getitem__(self, idx): return self.cont[idx], self.cat[idx], self.labels[idx]


# ------------------------------------------------------------
# Shared data preprocessing (mask included, mortality_1d target)
# ------------------------------------------------------------
def load_data_with_mask():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    input_cols = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]

    mask_columns = []
    for c in input_cols:
        mask_col = c + '_m'
        df[mask_col] = df[c].notna().astype(float)
        mask_columns.append(mask_col)

    df_train      = df[df['general_strat_fold'] < 18]
    train_medians = df_train[input_cols].median().to_dict()
    for c in [c for c, v in df_train[input_cols].isna().sum().items() if v > 0]:
        df.loc[df[c].isna(), c] = train_medians[c]
    df = df.copy()

    unique_counts = {c: len(np.unique(np.array(df[c]))) for c in input_cols}
    cat_features  = [c for c, v in unique_counts.items()
                     if v < 10 and not c.endswith("nan") and not c.startswith("labvalues")]
    cont_features = [c for c in input_cols if c not in cat_features]
    cont_features = cont_features + mask_columns

    df["vitals_acuity"] = df["vitals_acuity"].apply(lambda x: int(x) - 1)
    lbl_eth = ['demographics_ethnicity_asian','demographics_ethnicity_black/african',
               'demographics_ethnicity_hispanic/latino','demographics_ethnicity_other',
               'demographics_ethnicity_white']
    df["demographics_ethnicity"] = df.apply(lambda r: np.where([r[c] for c in lbl_eth])[0][0], axis=1)
    df.drop(lbl_eth, axis=1, inplace=True)
    ethnicity_masks = [c + '_m' for c in lbl_eth if (c + '_m') in df.columns]
    if ethnicity_masks:
        df.drop(ethnicity_masks, axis=1, inplace=True)
        mask_columns = [c for c in mask_columns if c not in ethnicity_masks]

    input_cols    = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]
    cat_features  = [c for c in input_cols if c in cat_features]
    cont_features = [c for c in input_cols if c not in cat_features]

    df["deterioration_mortality_1d"] = df["deterioration_mortality_1d"].replace(-999., np.nan)

    test_df = df[df['general_strat_fold'] == 19].reset_index(drop=True)
    test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)

    return test_df, cont_features, cat_features, unique_counts


test_df, cont_features, cat_features, unique_counts = load_data_with_mask()

shape   = ShapeCfg(static_dim=len(cont_features), static_dim_cat=len(cat_features))
mlp_cfg = MLPConfig(
    embedding_dims=[unique_counts[c] for c in cat_features],
    vocab_sizes=[unique_counts[c] for c in cat_features],
    lin_ftrs=[128, 128, 128]
)

test_loader = DataLoader(
    TabularDataset(test_df, cont_features, cat_features, "deterioration_mortality_1d"),
    batch_size=512, shuffle=False
)

# ==============================================================
# [1] MC Dropout: load checkpoint, run T=50 stochastic passes on test
# ==============================================================
print("Loading MC Dropout checkpoint...")
mc_model = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=1).to(DEVICE)
mc_model.load_state_dict(torch.load(MC_PT_PATH, map_location=DEVICE))

def mc_dropout_predict(loader, model, T=50):
    model.train()  # keep dropout active
    all_samples, all_labels = [], []
    with torch.no_grad():
        for cont, cat, labels in loader:
            cont, cat = cont.to(DEVICE), cat.to(DEVICE)
            batch_samples = []
            for _ in range(T):
                probs = torch.sigmoid(model(static=cont, static_cat=cat)["static"]).cpu().numpy()
                batch_samples.append(probs.squeeze(-1))
            all_samples.append(np.stack(batch_samples, axis=0))
            all_labels.append(labels.numpy().squeeze(-1))
    all_samples = np.concatenate(all_samples, axis=1)
    all_labels  = np.concatenate(all_labels,  axis=0)
    mean_probs = all_samples.mean(axis=0)
    p = mean_probs
    entropy = -(p * np.log(p + EPSILON) + (1 - p) * np.log(1 - p + EPSILON))
    return mean_probs, entropy, all_labels

mc_prob, mc_ent, mc_labels = mc_dropout_predict(test_loader, mc_model, MC_SAMPLES)

mc_mask = ~np.isnan(mc_labels)
mc_prob  = mc_prob[mc_mask]
mc_ent   = mc_ent[mc_mask]
mc_true  = mc_labels[mc_mask].astype(int)
mc_pred  = (mc_prob >= MC_THR).astype(int)

mc_tp_mask = (mc_pred == 1) & (mc_true == 1)
mc_fp_mask = (mc_pred == 1) & (mc_true == 0)
mc_tp_ent  = mc_ent[mc_tp_mask]
mc_fp_ent  = mc_ent[mc_fp_mask]
print(f"MC Dropout: TP n={mc_tp_mask.sum()}, FP n={mc_fp_mask.sum()}")

del mc_model
torch.cuda.empty_cache()

# ==============================================================
# [2] Deep Ensemble: load 5 members, compute entropy on test
# ==============================================================
print("Loading Deep Ensemble checkpoints...")
de_models = []
for pt_path in DE_PT_FILES:
    m = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=1).to(DEVICE)
    m.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    m.eval()
    de_models.append(m)

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
    stacked = np.stack(all_preds_per_model, axis=0)
    mean_pred = stacked.mean(axis=0)
    p = mean_pred
    entropy = -(p * np.log(p + EPSILON) + (1 - p) * np.log(1 - p + EPSILON))
    return mean_pred, entropy, all_labels

de_prob, de_ent, de_labels = ensemble_predict(de_models, test_loader)
de_prob = de_prob[:, 0]; de_ent = de_ent[:, 0]; de_labels = de_labels[:, 0]

de_mask = ~np.isnan(de_labels)
de_prob = de_prob[de_mask]
de_ent  = de_ent[de_mask]
de_true = de_labels[de_mask].astype(int)
de_pred = (de_prob >= DE_THR).astype(int)

de_tp_mask = (de_pred == 1) & (de_true == 1)
de_fp_mask = (de_pred == 1) & (de_true == 0)
de_tp_ent  = de_ent[de_tp_mask]
de_fp_ent  = de_ent[de_fp_mask]
print(f"Deep Ensemble: TP n={de_tp_mask.sum()}, FP n={de_fp_mask.sum()}")

del de_models
torch.cuda.empty_cache()

# ==============================================================
# Plot: TP vs FP uncertainty distributions (side by side)
# ==============================================================
shared_ymax = max(
    np.histogram(mc_tp_ent, bins=40)[0].max(),
    np.histogram(mc_fp_ent, bins=40)[0].max(),
    np.histogram(de_tp_ent, bins=40)[0].max(),
    np.histogram(de_fp_ent, bins=40)[0].max(),
) * 1.05  # small headroom

# --- MC Dropout plot ---
fig, ax = plt.subplots(figsize=(7, 5))
bins_mc = np.linspace(0, max(mc_tp_ent.max(), mc_fp_ent.max()), 40)
ax.hist(mc_tp_ent, bins=bins_mc, alpha=0.6, color='tab:blue',  label=f"TP (n={mc_tp_mask.sum()})")
ax.hist(mc_fp_ent, bins=bins_mc, alpha=0.6, color='tab:red',   label=f"FP (n={mc_fp_mask.sum()})")
ax.set_title(f"Uncertainty Distribution: TP vs FP — mortality_1d\n(MC Dropout, thr={MC_THR})")
ax.set_xlabel("Uncertainty (MC Dropout Entropy)")
ax.set_ylabel("Count")
ax.set_ylim(0, shared_ymax)
ax.legend()
plt.tight_layout()
fig_path_mc = os.path.join(OUT_DIR, "uncertainty_tp_fp_mortality1d_mcdropout.png")
plt.savefig(fig_path_mc, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path_mc}")

# --- Deep Ensemble plot ---
fig, ax = plt.subplots(figsize=(7, 5))
bins_de = np.linspace(0, max(de_tp_ent.max(), de_fp_ent.max()), 40)
ax.hist(de_tp_ent, bins=bins_de, alpha=0.6, color='tab:blue',  label=f"TP (n={de_tp_mask.sum()})")
ax.hist(de_fp_ent, bins=bins_de, alpha=0.6, color='tab:red',   label=f"FP (n={de_fp_mask.sum()})")
ax.set_title(f"Uncertainty Distribution: TP vs FP — mortality_1d\n(Deep Ensemble, thr={DE_THR})")
ax.set_xlabel("Uncertainty (Ensemble Entropy)")
ax.set_ylabel("Count")
ax.set_ylim(0, shared_ymax)
ax.legend()
plt.tight_layout()
fig_path_de = os.path.join(OUT_DIR, "uncertainty_tp_fp_mortality1d_deepensemble.png")
plt.savefig(fig_path_de, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path_de}")