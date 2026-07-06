"""
Deep Ensemble Base Model — Q-Model Feature Importance
ONLY for the best-performing configuration:
    Base Model : Deep Ensemble
    Threshold  : 0.10
    Strategy   : Simple
    Q-model    : XGBoost

이 스크립트는 sweep 코드(prob_threshold sweep for Deep Ensemble Q-model)와
완전히 동일한 데이터 파이프라인(mask 포함, train(fold 0~17) 기준 Q-model 학습,
icu_24h 단일 라벨)을 사용합니다. LR/MLP, CrossFit, 다른 threshold는
이미 성능 비교가 끝났으므로 여기서는 생략합니다.
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
from torch.utils.data import DataLoader, Dataset
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

# ============================================================
# Paths
# ============================================================
BASE_DIR    = "/user/gaad2403/MDS-ED/key/Final/DeepEnsemble"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
PNG_DIR     = os.path.join(RESULTS_DIR, "png")
DATA_PATH   = "/user/gaad2403/MDS-ED/src/data/memmap/mds_ed.csv"

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

# ============================================================
# Config — sweep 코드와 동일 + 최적 세팅 고정
# ============================================================
ICU24H_IDX   = 0                 # sweep과 동일 (라벨이 icu_24h 하나뿐)
BATCH_SIZE   = 32
LIN_FTRS     = [128, 128, 128]
M            = 5
EPSILON      = 1e-10
RANDOM_STATE = 42
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENSEMBLE_FEATURE_COLS = ["prob_icu24h", "variance", "entropy", "spread"]
THRESHOLD_FOR_IMPORTANCE = 0.10   # ✅ 최적 threshold (FP+FN 절대값 기준 확정값)

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

print(f"Ensemble features: {ENSEMBLE_FEATURE_COLS}")
print(f"Device: {DEVICE}")
print(f"Target config: Deep Ensemble | thr={THRESHOLD_FOR_IMPORTANCE} | Simple | XGB")

# ============================================================
# Model definitions (sweep 코드와 동일 — base model 구조 변경 없음)
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

# ============================================================
# 1. Load & preprocess data (✅ mask 포함, sweep 코드와 동일)
# ============================================================
print("\nLoading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

input_cols = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]

# ✅ mask 컬럼 추가 (sweep 코드와 동일)
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

# ✅ icu_24h 단일 라벨 (sweep 코드와 동일)
lbl_itos = ["icu_24h"]
for c in lbl_itos:
    df["deterioration_" + c] = df["deterioration_" + c].replace(-999., np.nan)

# ✅ train(fold 0~17) 기준 (sweep 코드와 동일 — val 아님)
train_df = df[df['general_strat_fold'].isin(range(0, 18))].reset_index(drop=True)
test_df  = df[df['general_strat_fold'] == 19].reset_index(drop=True)
train_df = train_df[train_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df  = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"cont_features (mask 포함): {len(cont_features)}, cat_features: {len(cat_features)}")

# ============================================================
# 2. Dataset & DataLoader
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, df, cont_f, cat_f, lbl_cols):
        self.cont   = torch.tensor(df[cont_f].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_f].values,  dtype=torch.long)
        self.labels = torch.tensor(df[["deterioration_"+c for c in lbl_cols]].values, dtype=torch.float32)
    def __len__(self): return len(self.cont)
    def __getitem__(self, i): return self.cont[i], self.cat[i], self.labels[i]

train_loader = DataLoader(TabularDataset(train_df, cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(TabularDataset(test_df, cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================
# 3. Load ensemble members (✅ mask 버전 checkpoint) & inference
# ============================================================
shape   = ShapeCfg(static_dim=len(cont_features), static_dim_cat=len(cat_features))
mlp_cfg = MLPConfig(
    embedding_dims=[unique_counts[c] for c in cat_features],
    vocab_sizes=[unique_counts[c] for c in cat_features],
    lin_ftrs=LIN_FTRS
)

print(f"\nLoading {M} ensemble members (mask version)...")
ensemble_models = []
for m in range(M):
    pt_path = os.path.join(BASE_DIR, f"ensemble_member_{m}_icu24h_only_mask.pt")
    model   = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=len(lbl_itos)).to(DEVICE)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
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
    mean_pred = stacked.mean(axis=0)
    variance  = stacked.var(axis=0)
    spread    = stacked.max(axis=0) - stacked.min(axis=0)
    p         = mean_pred
    entropy   = -(p * np.log(p + EPSILON) + (1-p) * np.log(1-p + EPSILON))

    return mean_pred, variance, entropy, spread, all_labels


print("Running ensemble inference...")
print("  Train set...")
train_mean, train_var, train_ent, train_spr, train_labels = ensemble_predict(ensemble_models, train_loader)
print("  Test set...")
test_mean, test_var, test_ent, test_spr, test_labels = ensemble_predict(ensemble_models, test_loader)
print("  Done.")

train_mask = ~np.isnan(train_labels[:, ICU24H_IDX])
test_mask  = ~np.isnan(test_labels[:,  ICU24H_IDX])

train_prob_icu = train_mean[train_mask,   ICU24H_IDX]
train_var_icu  = train_var[train_mask,    ICU24H_IDX]
train_ent_icu  = train_ent[train_mask,    ICU24H_IDX]
train_spr_icu  = train_spr[train_mask,    ICU24H_IDX]
train_true_icu = train_labels[train_mask, ICU24H_IDX].astype(int)

test_prob_icu = test_mean[test_mask,   ICU24H_IDX]
test_var_icu  = test_var[test_mask,    ICU24H_IDX]
test_ent_icu  = test_ent[test_mask,    ICU24H_IDX]
test_spr_icu  = test_spr[test_mask,    ICU24H_IDX]
test_true_icu = test_labels[test_mask, ICU24H_IDX].astype(int)

print(f"Train ICU samples: {len(train_prob_icu)}")
print(f"Test  ICU samples: {len(test_prob_icu)}")

# ============================================================
# Prepare Q-Model features: Base features + Ensemble stats (TRAIN 기준)
# ============================================================
train_df_masked = train_df[train_mask].reset_index(drop=True)
test_df_masked  = test_df[test_mask].reset_index(drop=True)

X_train_features = np.hstack([
    train_df_masked[cont_features].values.astype(np.float32),
    train_df_masked[cat_features].values.astype(np.float32)
])
X_test_features = np.hstack([
    test_df_masked[cont_features].values.astype(np.float32),
    test_df_masked[cat_features].values.astype(np.float32)
])

feature_names_qmodel = cont_features + cat_features + ENSEMBLE_FEATURE_COLS
print(f"\n📊 Q-Model features: {len(feature_names_qmodel)} "
      f"(cont+mask {len(cont_features)} + cat {len(cat_features)} + ensemble stats 4)")

X_train_q_combined = np.hstack([
    X_train_features,
    train_prob_icu.reshape(-1, 1),
    train_var_icu.reshape(-1, 1),
    train_ent_icu.reshape(-1, 1),
    train_spr_icu.reshape(-1, 1)
]).astype(np.float32)

X_test_q_combined = np.hstack([
    X_test_features,
    test_prob_icu.reshape(-1, 1),
    test_var_icu.reshape(-1, 1),
    test_ent_icu.reshape(-1, 1),
    test_spr_icu.reshape(-1, 1)
]).astype(np.float32)

# ============================================================
# Error labels @ threshold=0.10 (TRAIN 기준, sweep과 동일 정의)
# ============================================================
train_pred_final = (train_prob_icu >= THRESHOLD_FOR_IMPORTANCE).astype(int)
test_pred_final  = (test_prob_icu  >= THRESHOLD_FOR_IMPORTANCE).astype(int)
train_err_final  = (train_pred_final != train_true_icu).astype(int)
test_err_final   = (test_pred_final  != test_true_icu).astype(int)

print(f"\n✅ Using threshold={THRESHOLD_FOR_IMPORTANCE:.2f}")
print(f"   Train: {len(train_err_final)} samples, {train_err_final.sum()} errors")
print(f"   Test:  {len(test_err_final)} samples, {test_err_final.sum()} errors")

# ============================================================
# XGB Q-model (Simple strategy) — feature importance
# ============================================================
print(f"\n[XGB / Simple] Training & extracting feature importance...")
model_xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          eval_metric='logloss', random_state=RANDOM_STATE, verbosity=0)
model_xgb.fit(X_train_q_combined, train_err_final)

importance_dict  = model_xgb.get_booster().get_score(importance_type='gain')
importances_xgb  = np.array([importance_dict.get(f"f{i}", 0.0) for i in range(len(feature_names_qmodel))])

test_probs_xgb = model_xgb.predict_proba(X_test_q_combined)[:, 1]
test_auroc_xgb = roc_auc_score(test_err_final, test_probs_xgb) if len(np.unique(test_err_final)) > 1 else np.nan
print(f"  ✅ Test AUROC (error prediction): {test_auroc_xgb:.4f}")

print(f"  Top 15 features (XGB gain):")
top_idx = np.argsort(importances_xgb)[::-1][:15]
for rank, idx in enumerate(top_idx, 1):
    print(f"    {rank:2d}. {feature_names_qmodel[idx]:40s} {importances_xgb[idx]:8.4f}")

# ============================================================
# Save feature importance CSV
# ============================================================
importance_df = pd.DataFrame({
    'feature': feature_names_qmodel,
    'XGB_importance_gain': importances_xgb
}).sort_values('XGB_importance_gain', ascending=False)

importance_path = os.path.join(CSV_DIR, "qmodel_feature_importance_deepensemble_thr010_simple_xgb.csv")
importance_df.to_csv(importance_path, index=False)
print(f"\n✓ Feature importance saved: {importance_path}")

# ============================================================
# Plot: Top-5 feature importance (single panel)
# ============================================================
top5 = importance_df.nlargest(5, 'XGB_importance_gain')

fig, ax = plt.subplots(figsize=(9, 5))
y_pos  = np.arange(len(top5))
BAR_COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"] 
BAR_HEIGHT = 0.5        # 두께 조정 (기본 0.8보다 살짝 얇게)
ax.barh(y_pos, top5['XGB_importance_gain'].values, height=BAR_HEIGHT, color=BAR_COLORS)
ax.set_yticks(y_pos)
ax.set_yticklabels(top5['feature'].values, fontsize=11)
ax.invert_yaxis()
ax.set_xlabel("XGBoost Importance (gain)", fontsize=11, fontweight='bold')
ax.set_title(
    f"Q-Model Feature Importance — Top 5\n"
    f"(Deep Ensemble, threshold={THRESHOLD_FOR_IMPORTANCE:.2f}, Simple, XGB)",
    fontsize=13, fontweight='bold'
)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
fig_path = os.path.join(PNG_DIR, "qmodel_feature_importance_deepensemble_thr010_simple_xgb_top5.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Plot saved: {fig_path}")

print("\n" + "=" * 70)
print("✅ All done! (Deep Ensemble | thr=0.10 | Simple | XGB only)")
print("=" * 70)
print(f"  Feature Importance CSV: {importance_path}")
print(f"  Plot:                   {fig_path}")
print("=" * 70)