"""
Q-model feature importance comparison across the four base models (top-5, single plot).
Target: 24h mortality (mortality_1d).

Confirmed (threshold, strategy, Q-model) per base model -- picked as the
specificity-maximizing point within the sens>=0.80 region for each model:
    BasicMLP      : thr=0.020,  Simple,   XGB
    MC Dropout    : thr=0.010,  Simple,   XGB
    Deep Ensemble : thr=0.018,  Simple,   MLP
    XGBoost       : thr=0.0008, CrossFit, MLP

XGB models use gain importance; MLP models use SHAP (DeepExplainer). Each model's
importances are normalized to max=1.0 before combining into a single comparison plot.

Outputs:
    - feature_importance_{model}.csv per base model (all features)
    - combined top-5 comparison plot (png)
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import StratifiedKFold
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

RANDOM_STATE = 42
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_FOLDS      = 5
DATA_PATH    = "/user/gaad2403/MDS-ED/src/data/memmap/mds_ed.csv"

COMPARISON_DIR = "/user/gaad2403/MDS-ED/key/Final/modality/comparison"
os.makedirs(COMPARISON_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

XGB_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                  eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0)

# Q-model MLP hyperparameters (same as the sweep scripts)
MLP_HIDDEN   = [64, 32]
MLP_EPOCHS   = 50
MLP_LR       = 1e-3
MLP_BATCH    = 64
MLP_DROPOUT  = 0.3


# ------------------------------------------------------------
# Shared torch encoder (BasicMLP / MC Dropout / Deep Ensemble)
# ------------------------------------------------------------
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
class BasicEncoderStaticConfig(EncoderStaticBaseConfig):
    _target_: str = "clinical_ts.tabular.base.BasicEncoderStatic"
    embedding_dims: List[int] = field(default_factory=list)
    vocab_sizes: List[int]    = field(default_factory=list)

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
    def __init__(self, df, cont_f, cat_f, lbl_cols):
        self.cont   = torch.tensor(df[cont_f].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_f].values,  dtype=torch.long)
        self.labels = torch.tensor(df[["deterioration_"+c for c in lbl_cols]].values, dtype=torch.float32)
    def __len__(self): return len(self.cont)
    def __getitem__(self, i): return self.cont[i], self.cat[i], self.labels[i]


# ------------------------------------------------------------
# Q-model MLP
# ------------------------------------------------------------
class QModelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        # keep (batch, 1) -- shap.DeepExplainer indexes outputs.shape[1]
        return self.net(x)


def train_qmodel_mlp(X_tr, y_tr):
    model   = QModelMLP(X_tr.shape[1], MLP_HIDDEN, MLP_DROPOUT).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                  torch.tensor(y_tr.astype(np.float32), dtype=torch.float32)),
                    batch_size=MLP_BATCH, shuffle=True)
    for _ in range(MLP_EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(torch.sigmoid(model(xb)).squeeze(-1), yb).backward()
            opt.step()
    model.eval()
    return model


def mlp_importance_shap(model, X_all, n_background=100, n_sample=200):
    """Mean |SHAP value| per feature for a trained Q-model MLP."""
    import shap
    bg_idx = np.random.choice(len(X_all), size=min(n_background, len(X_all)//2), replace=False)
    background = torch.tensor(X_all[bg_idx], dtype=torch.float32).to(DEVICE)
    explainer = shap.DeepExplainer(model, background)

    sample_idx = np.random.choice(len(X_all), size=min(n_sample, len(X_all)//2), replace=False)
    sample = torch.tensor(X_all[sample_idx], dtype=torch.float32).to(DEVICE)

    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)
    return np.mean(np.abs(shap_values), axis=0)


def mlp_importance_simple(X_tr, y_tr, feature_names):
    """Simple strategy: one MLP trained on the full train set, SHAP importance."""
    model = train_qmodel_mlp(X_tr, y_tr)
    return mlp_importance_shap(model, X_tr)


def mlp_importance_crossfit(X_tr, y_tr, feature_names, n_folds=N_FOLDS):
    """CrossFit strategy: average SHAP importance across 5 fold-trained MLPs."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    all_importances = []
    for tri, vli in skf.split(X_tr, y_tr):
        model = train_qmodel_mlp(X_tr[tri], y_tr[tri])
        imp = mlp_importance_shap(model, X_tr[vli])
        all_importances.append(imp)
    return np.mean(all_importances, axis=0)


def xgb_importance_simple(X_tr, y_tr, feature_names):
    """Simple strategy: gain importance of one XGB trained on the full train set."""
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr, y_tr)
    imp_dict = model.get_booster().get_score(importance_type='gain')
    return np.array([imp_dict.get(f"f{i}", 0.0) for i in range(len(feature_names))])


def xgb_importance_crossfit(X_tr, y_tr, feature_names, n_folds=N_FOLDS):
    """CrossFit strategy: average gain importance across 5 fold-trained XGBs."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    all_importances = []
    for tri, _ in skf.split(X_tr, y_tr):
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr[tri], y_tr[tri])
        imp_dict = model.get_booster().get_score(importance_type='gain')
        importances = np.array([imp_dict.get(f"f{i}", 0.0) for i in range(len(feature_names))])
        all_importances.append(importances)
    return np.mean(all_importances, axis=0)


# ------------------------------------------------------------
# Shared preprocessing (mask columns included for all 4 base models)
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

    lbl_itos = ["mortality_1d"]
    for c in lbl_itos:
        df["deterioration_" + c] = df["deterioration_" + c].replace(-999., np.nan)

    train_df = df[df['general_strat_fold'].isin(range(0, 18))].reset_index(drop=True)
    test_df  = df[df['general_strat_fold'] == 19].reset_index(drop=True)
    train_df = train_df[train_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
    test_df  = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)

    return df, train_df, test_df, cont_features, cat_features, unique_counts, lbl_itos


def get_probs_labels(loader, encoder):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for cont, cat, labels in loader:
            cont, cat = cont.to(DEVICE), cat.to(DEVICE)
            probs = torch.sigmoid(encoder(static=cont, static_cat=cat)["static"])
            all_probs.append(probs.cpu().numpy()); all_labels.append(labels.numpy())
    return np.concatenate(all_probs,0), np.concatenate(all_labels,0)


all_top5 = {}  # model_label -> DataFrame(feature, importance_norm)

# ==============================================================
# [1] BasicMLP | threshold=0.020 | Simple | XGB
# ==============================================================
print("BasicMLP | threshold=0.020 | Simple | XGB")

BASICMLP_BASE_DIR = "/fs/dss/home/gaad2403/MDS-ED/key/Final/modality"
BASICMLP_PT_PATH  = os.path.join(BASICMLP_BASE_DIR, "best_basicmlp_mortality1d_only.pt")
BASICMLP_THR      = 0.020

df_b, train_df_b, test_df_b, cont_b, cat_b, uniq_b, lbl_b = load_data_with_mask()

shape_b   = ShapeCfg(static_dim=len(cont_b), static_dim_cat=len(cat_b))
mlpcfg_b  = MLPConfig(embedding_dims=[uniq_b[c] for c in cat_b],
                      vocab_sizes=[uniq_b[c] for c in cat_b], lin_ftrs=[128,128,128])
enc_b = BasicEncoderStaticMLP(mlpcfg_b, shape_b, target_dim=1).to(DEVICE)
enc_b.load_state_dict(torch.load(BASICMLP_PT_PATH, map_location=DEVICE))
enc_b.eval()

train_loader_b = DataLoader(TabularDataset(train_df_b, cont_b, cat_b, lbl_b), batch_size=512, shuffle=False)

train_probs_b, train_labels_b = get_probs_labels(train_loader_b, enc_b)
train_prob_b  = train_probs_b[:, 0]
train_true_b  = train_labels_b[:, 0]
mask_b        = ~np.isnan(train_true_b)
train_prob_b  = train_prob_b[mask_b]
train_true_b  = train_true_b[mask_b].astype(int)
train_df_masked_b = train_df_b[mask_b].reset_index(drop=True)

X_train_feat_b = np.hstack([
    train_df_masked_b[cont_b].values.astype(np.float32),
    train_df_masked_b[cat_b].values.astype(np.float32)
])
feature_names_b = cont_b + cat_b + ["base_model_prob_mortality1d"]
X_train_q_b = np.hstack([X_train_feat_b, train_prob_b.reshape(-1,1)]).astype(np.float32)

train_pred_b = (train_prob_b >= BASICMLP_THR).astype(int)
train_err_b  = (train_pred_b != train_true_b).astype(int)

imp_b = xgb_importance_simple(X_train_q_b, train_err_b, feature_names_b)
imp_df_b = pd.DataFrame({'feature': feature_names_b, 'importance': imp_b}).sort_values('importance', ascending=False)
imp_df_b.to_csv(os.path.join(COMPARISON_DIR, "feature_importance_basicmlp.csv"), index=False)
print(imp_df_b.head(5).to_string(index=False))

top5_b = imp_df_b.nlargest(5, 'importance').copy()
top5_b['importance_norm'] = top5_b['importance'] / top5_b['importance'].max()
all_top5['BasicMLP'] = top5_b[['feature', 'importance_norm']]

del enc_b, df_b, train_df_b, test_df_b
torch.cuda.empty_cache()

# ==============================================================
# [2] Deep Ensemble | threshold=0.018 | Simple | MLP
# ==============================================================
print("Deep Ensemble | threshold=0.018 | Simple | MLP")

DEEPENS_BASE_DIR = "/fs/dss/home/gaad2403/MDS-ED/key/Final/modality"
DEEPENS_THR      = 0.018
M = 5

df_d, train_df_d, test_df_d, cont_d, cat_d, uniq_d, lbl_d = load_data_with_mask()

shape_d  = ShapeCfg(static_dim=len(cont_d), static_dim_cat=len(cat_d))
mlpcfg_d = MLPConfig(embedding_dims=[uniq_d[c] for c in cat_d],
                     vocab_sizes=[uniq_d[c] for c in cat_d], lin_ftrs=[128,128,128])

ensemble_models_d = []
for m in range(M):
    pt_path = os.path.join(DEEPENS_BASE_DIR, f"ensemble_member_{m}_mortality1d_only_mask.pt")
    model = BasicEncoderStaticMLP(mlpcfg_d, shape_d, target_dim=1).to(DEVICE)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    model.eval()
    ensemble_models_d.append(model)

train_loader_d = DataLoader(TabularDataset(train_df_d, cont_d, cat_d, lbl_d), batch_size=512, shuffle=False)

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
    variance  = stacked.var(axis=0)
    spread    = stacked.max(axis=0) - stacked.min(axis=0)
    p = mean_pred
    entropy = -(p*np.log(p+1e-10) + (1-p)*np.log(1-p+1e-10))
    return mean_pred, variance, entropy, spread, all_labels

train_mean_d, train_var_d, train_ent_d, train_spr_d, train_labels_d = ensemble_predict(ensemble_models_d, train_loader_d)

train_mask_d = ~np.isnan(train_labels_d[:, 0])
train_prob_d = train_mean_d[train_mask_d, 0]
train_var_d2 = train_var_d[train_mask_d, 0]
train_ent_d2 = train_ent_d[train_mask_d, 0]
train_spr_d2 = train_spr_d[train_mask_d, 0]
train_true_d = train_labels_d[train_mask_d, 0].astype(int)
train_df_masked_d = train_df_d[train_mask_d].reset_index(drop=True)

X_train_feat_d = np.hstack([
    train_df_masked_d[cont_d].values.astype(np.float32),
    train_df_masked_d[cat_d].values.astype(np.float32)
])
feature_names_d = cont_d + cat_d + ["prob_mortality1d", "variance", "entropy", "spread"]
X_train_q_d = np.hstack([
    X_train_feat_d, train_prob_d.reshape(-1,1), train_var_d2.reshape(-1,1),
    train_ent_d2.reshape(-1,1), train_spr_d2.reshape(-1,1)
]).astype(np.float32)

train_pred_d = (train_prob_d >= DEEPENS_THR).astype(int)
train_err_d  = (train_pred_d != train_true_d).astype(int)

print("  Computing SHAP importance for MLP (Simple, full train set)...")
imp_d = mlp_importance_simple(X_train_q_d, train_err_d, feature_names_d)
imp_df_d = pd.DataFrame({'feature': feature_names_d, 'importance': imp_d}).sort_values('importance', ascending=False)
imp_df_d.to_csv(os.path.join(COMPARISON_DIR, "feature_importance_deepensemble.csv"), index=False)
print(imp_df_d.head(5).to_string(index=False))

top5_d = imp_df_d.nlargest(5, 'importance').copy()
top5_d['importance_norm'] = top5_d['importance'] / top5_d['importance'].max()
all_top5['Deep Ensemble'] = top5_d[['feature', 'importance_norm']]

del ensemble_models_d, df_d, train_df_d, test_df_d
torch.cuda.empty_cache()

# ==============================================================
# [3] MC Dropout | threshold=0.010 | Simple | XGB
# ==============================================================
print("MC Dropout | threshold=0.010 | Simple | XGB")

MCDROPOUT_BASE_DIR = "/fs/dss/home/gaad2403/MDS-ED/key/Final/modality"
MCDROPOUT_PT_PATH  = os.path.join(MCDROPOUT_BASE_DIR, "best_mcdropout_mortality1d_only_mask.pt")
MCDROPOUT_THR      = 0.010
MC_SAMPLES         = 50
EPSILON            = 1e-10

df_m, train_df_m, test_df_m, cont_m, cat_m, uniq_m, lbl_m = load_data_with_mask()

shape_m  = ShapeCfg(static_dim=len(cont_m), static_dim_cat=len(cat_m))
mlpcfg_m = MLPConfig(embedding_dims=[uniq_m[c] for c in cat_m],
                     vocab_sizes=[uniq_m[c] for c in cat_m], lin_ftrs=[128,128,128])
enc_m = BasicEncoderStaticMLP(mlpcfg_m, shape_m, target_dim=1).to(DEVICE)
enc_m.load_state_dict(torch.load(MCDROPOUT_PT_PATH, map_location=DEVICE))

train_loader_m = DataLoader(TabularDataset(train_df_m, cont_m, cat_m, lbl_m), batch_size=512, shuffle=False)

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
    variance   = all_samples.var(axis=0)
    p = mean_probs
    entropy = -(p*np.log(p+EPSILON) + (1-p)*np.log(1-p+EPSILON))
    return mean_probs, variance, entropy, all_labels

train_mean_m, train_var_m, train_ent_m, train_labels_m = mc_dropout_predict(train_loader_m, enc_m, MC_SAMPLES)

train_mask_m  = ~np.isnan(train_labels_m)
train_prob_m  = train_mean_m[train_mask_m]
train_var_m2  = train_var_m[train_mask_m]
train_ent_m2  = train_ent_m[train_mask_m]
train_true_m  = train_labels_m[train_mask_m].astype(int)
train_df_masked_m = train_df_m[train_mask_m].reset_index(drop=True)

X_train_feat_m = np.hstack([
    train_df_masked_m[cont_m].values.astype(np.float32),
    train_df_masked_m[cat_m].values.astype(np.float32)
])
feature_names_m = cont_m + cat_m + ["prob_mortality1d", "variance", "entropy"]
X_train_q_m = np.hstack([
    X_train_feat_m, train_prob_m.reshape(-1,1), train_var_m2.reshape(-1,1), train_ent_m2.reshape(-1,1)
]).astype(np.float32)

train_pred_m = (train_prob_m >= MCDROPOUT_THR).astype(int)
train_err_m  = (train_pred_m != train_true_m).astype(int)

imp_m = xgb_importance_simple(X_train_q_m, train_err_m, feature_names_m)
imp_df_m = pd.DataFrame({'feature': feature_names_m, 'importance': imp_m}).sort_values('importance', ascending=False)
imp_df_m.to_csv(os.path.join(COMPARISON_DIR, "feature_importance_mcdropout.csv"), index=False)
print(imp_df_m.head(5).to_string(index=False))

top5_m = imp_df_m.nlargest(5, 'importance').copy()
top5_m['importance_norm'] = top5_m['importance'] / top5_m['importance'].max()
all_top5['MC Dropout'] = top5_m[['feature', 'importance_norm']]

del enc_m, df_m, train_df_m, test_df_m
torch.cuda.empty_cache()

# ==============================================================
# [4] XGBoost (base model) | threshold=0.0008 | CrossFit | MLP
# ==============================================================
print("XGBoost | threshold=0.0008 | CrossFit | MLP (mask included)")

XGB_THR = 0.0008

df_x, train_df_x, test_df_x, cont_x, cat_x, uniq_x, lbl_x = load_data_with_mask()

x_train_x = np.hstack([
    train_df_x[cont_x].values.astype(np.float32),
    train_df_x[cat_x].values.astype(np.float32)
])
y_train_x = train_df_x["deterioration_mortality_1d"].values

mask_tr_x = ~np.isnan(y_train_x)
x_tr_x = x_train_x[mask_tr_x]
y_tr_x = y_train_x[mask_tr_x].astype(int)

base_model_x = XGBClassifier(random_state=RANDOM_STATE, n_jobs=4, eval_metric='logloss')
base_model_x.fit(x_tr_x, y_tr_x)

train_prob_x = base_model_x.predict_proba(x_tr_x)[:, 1]
train_true_x = y_tr_x

feature_names_x = cont_x + cat_x + ["base_model_prob_mortality1d"]
X_train_q_x = np.hstack([x_tr_x, train_prob_x.reshape(-1,1)]).astype(np.float32)

train_pred_x = (train_prob_x >= XGB_THR).astype(int)
train_err_x  = (train_pred_x != train_true_x).astype(int)

print("  Computing SHAP importance for MLP (CrossFit, 5-fold)...")
imp_x = mlp_importance_crossfit(X_train_q_x, train_err_x, feature_names_x)
imp_df_x = pd.DataFrame({'feature': feature_names_x, 'importance': imp_x}).sort_values('importance', ascending=False)
imp_df_x.to_csv(os.path.join(COMPARISON_DIR, "feature_importance_xgboost.csv"), index=False)
print(imp_df_x.head(5).to_string(index=False))

top5_x = imp_df_x.nlargest(5, 'importance').copy()
top5_x['importance_norm'] = top5_x['importance'] / top5_x['importance'].max()
all_top5['XGBoost'] = top5_x[['feature', 'importance_norm']]

# ==============================================================
# Combined comparison plot: top-5 features across all 4 models
# Note: BasicMLP/MC Dropout use XGB gain, Deep Ensemble/XGBoost use MLP SHAP.
# Different scales, so each model is normalized to max=1.0 before plotting.
# ==============================================================
print("Building combined comparison plot")

MODEL_COLORS = {
    'BasicMLP':      '#1f77b4',
    'MC Dropout':    '#2ca02c',
    'Deep Ensemble': '#ff7f0e',
    'XGBoost':       '#d62728',
}
MODEL_ORDER = ['BasicMLP', 'MC Dropout', 'Deep Ensemble', 'XGBoost']
IMPORTANCE_METHOD = {
    'BasicMLP':      'XGB gain (Simple)',
    'MC Dropout':    'XGB gain (Simple)',
    'Deep Ensemble': 'MLP SHAP (Simple)',
    'XGBoost':       'MLP SHAP (CrossFit)',
}

union_features = list(dict.fromkeys(
    f for model in MODEL_ORDER for f in all_top5[model]['feature']
))

def lookup(model, feat):
    row = all_top5[model][all_top5[model]['feature'] == feat]
    return row['importance_norm'].values[0] if len(row) > 0 else 0.0

n_models = len(MODEL_ORDER)
bar_height = 0.8 / n_models
y = np.arange(len(union_features))

fig, ax = plt.subplots(figsize=(10, max(4, len(union_features) * 0.6)))
for i, model in enumerate(MODEL_ORDER):
    offset = (i - (n_models - 1) / 2) * bar_height
    vals = [lookup(model, f) for f in union_features]
    ax.barh(y + offset, vals, height=bar_height,
            label=f"{model} ({IMPORTANCE_METHOD[model]})", color=MODEL_COLORS[model])

ax.set_yticks(y)
ax.set_yticklabels(union_features, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Normalized Importance (within-model max = 1.0)", fontsize=11, fontweight='bold')
ax.set_title("Q-Model Feature Importance - Top 5 per Base Model (24h Mortality)\n"
             "(each at its confirmed best threshold/strategy; "
             "XGB=gain importance, MLP=SHAP importance)",
             fontsize=12, fontweight='bold')
ax.legend(title="Base Model (Q-model importance method)", fontsize=8)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig_path = os.path.join(COMPARISON_DIR, "qmodel_feature_importance_top5_4models_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Combined plot saved: {fig_path}")