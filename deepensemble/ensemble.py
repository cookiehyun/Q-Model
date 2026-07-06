"""
Deep Ensemble Base Model — ICU-24h ONLY + mask 포함 (교수님 지침)
=====================================================================
변경: 원본 ensemble_icu24h_only.py에 mask 컬럼 추가.
      (딥러닝 base 모델은 원본+mask 사용 — 교수님 demo.ipynb 지침)
      mask는 notna() 기준 (1=값 있음, 0=결측).
      체크포인트 파일명: ensemble_member_{m}_icu24h_only_mask.pt
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
from sklearn.metrics import roc_auc_score
from clinical_ts.template_modules import EncoderStaticBase, EncoderStaticBaseConfig
from collections.abc import Iterable
from clinical_ts.ts.basic_conv1d_modules.basic_conv1d import bn_drop_lin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Model Definition (원본과 동일)
# ============================================================
class BasicEncoderStatic(EncoderStaticBase):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        self.input_channels_cat  = hparams_input_shape.static_dim_cat
        self.input_channels_cont = hparams_input_shape.static_dim
        assert (len(hparams_encoder_static.embedding_dims) == hparams_input_shape.static_dim_cat
                and len(hparams_encoder_static.vocab_sizes) == hparams_input_shape.static_dim_cat)
        self.embeddings = nn.ModuleList()
        for v, e in zip(hparams_encoder_static.vocab_sizes, hparams_encoder_static.embedding_dims):
            self.embeddings.append(nn.Embedding(v, e))
        self.input_dim      = int(np.sum(hparams_encoder_static.embedding_dims) + hparams_input_shape.static_dim)
        self.input_channels = hparams_input_shape.static_dim + hparams_input_shape.static_dim_cat

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

    def forward(self, **kwargs):      raise NotImplementedError
    def get_output_shape(self):       raise NotImplementedError


@dataclass
class BasicEncoderStaticConfig(EncoderStaticBaseConfig):
    _target_: str = "clinical_ts.tabular.base.BasicEncoderStatic"
    embedding_dims: List[int] = field(default_factory=lambda: [])
    vocab_sizes:    List[int] = field(default_factory=lambda: [])


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
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
        layers = []
        for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
            layers += bn_drop_lin(ni, no, hparams_encoder_static.batch_norm, p, actn, layer_norm=False)
        self.layers = nn.Sequential(*layers)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.static_dim     = int(lin_ftrs[-1])
        self.output_shape.static_dim_cat = 0

    def forward(self, **kwargs):
        res = self.embed(**kwargs)
        return {"static": self.layers(res)}

    def get_output_shape(self):
        return self.output_shape


@dataclass
class BasicEncoderStaticMLPConfig(BasicEncoderStaticConfig):
    _target_: str = "clinical_ts.tabular.base.BasicEncoderStaticMLP"
    lin_ftrs:   List[int] = field(default_factory=lambda: [512])
    dropout:    float = 0.5
    batch_norm: bool  = True


# ============================================================
# Config
# ============================================================
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 0.001
WEIGHT_DECAY = 0.001
LIN_FTRS     = [128, 128, 128]
M            = 5
PROB_THRESHOLD = 0.10
EPSILON      = 1e-10
PERCENTILES  = [75, 90, 95]
TARGET_TASK = "icu_24h"

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
PNG_DIR     = os.path.join(RESULTS_DIR, "png")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)
print(f"Results → {RESULTS_DIR}")

DATA_PATH = "/user/gaad2403/MDS-ED/src/data/memmap/mds_ed.csv"

# ============================================================
# 1. Load data
# ============================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"shape: {df.shape}")

input_cols    = [c for c in df.columns if c.split("_")[0] in ['biometrics', 'demographics', 'labvalues', 'vitals']]

# ✅ [교수님 원안] 딥러닝 모델은 mask 컬럼 포함 (notna 기준: 1=값 있음, 0=결측)
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
cont_features = cont_features + mask_columns  # ✅ mask 추가

df["vitals_acuity"] = df["vitals_acuity"].apply(lambda x: int(x) - 1)

lbl_itos_ethnicity = [
    'demographics_ethnicity_asian',
    'demographics_ethnicity_black/african',
    'demographics_ethnicity_hispanic/latino',
    'demographics_ethnicity_other',
    'demographics_ethnicity_white'
]
df["demographics_ethnicity"] = df.apply(
    lambda row: np.where([row[c] for c in lbl_itos_ethnicity])[0][0], axis=1
)
df.drop(lbl_itos_ethnicity, axis=1, inplace=True)
# ✅ ethnicity 원-핫이 병합되며 사라지므로 대응 mask도 제거
ethnicity_masks = [c + '_m' for c in lbl_itos_ethnicity if (c + '_m') in df.columns]
if ethnicity_masks:
    df.drop(ethnicity_masks, axis=1, inplace=True)
    mask_columns = [c for c in mask_columns if c not in ethnicity_masks]

input_cols    = [c for c in df.columns if c.split("_")[0] in ['biometrics', 'demographics', 'labvalues', 'vitals']]
cat_features  = [c for c in input_cols if c in cat_features]
cont_features = [c for c in input_cols if c not in cat_features]  # mask 자동 재포함됨

df["deterioration_" + TARGET_TASK] = df["deterioration_" + TARGET_TASK].replace(-999., np.nan)

print(f"categorical: {len(cat_features)}, continuous (mask 포함): {len(cont_features)}")

# ============================================================
# 2. Split
# ============================================================
train_df = df[df['general_strat_fold'].isin(range(0, 18))].reset_index(drop=True)
val_df   = df[df['general_strat_fold'] == 18].reset_index(drop=True)
test_df  = df[df['general_strat_fold'] == 19].reset_index(drop=True)

val_df  = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================
# 3. Dataset
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, df, cont_features, cat_features, target_task):
        self.cont   = torch.tensor(df[cont_features].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_features].values,  dtype=torch.long)
        self.labels = torch.tensor(df[["deterioration_" + target_task]].values, dtype=torch.float32)
    def __len__(self): return len(self.cont)
    def __getitem__(self, idx): return self.cont[idx], self.cat[idx], self.labels[idx]

train_ds = TabularDataset(train_df, cont_features, cat_features, TARGET_TASK)
val_ds   = TabularDataset(val_df,   cont_features, cat_features, TARGET_TASK)
test_ds  = TabularDataset(test_df,  cont_features, cat_features, TARGET_TASK)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================
# 4. Model config
# ============================================================
@dataclass
class MLPConfig:
    embedding_dims: List[int] = field(default_factory=lambda: [])
    vocab_sizes:    List[int] = field(default_factory=lambda: [])
    lin_ftrs:       List[int] = field(default_factory=lambda: [128, 128, 128])
    dropout:    float = 0.5
    batch_norm: bool  = True

@dataclass
class ShapeCfg:
    static_dim: int = 0; static_dim_cat: int = 0; channels: int = 0
    length: int = 0;     sequence_last: bool = False; channels2: int = 0

shape   = ShapeCfg(static_dim=len(cont_features), static_dim_cat=len(cat_features))
mlp_cfg = MLPConfig(
    embedding_dims=[unique_counts[c] for c in cat_features],
    vocab_sizes=[unique_counts[c]    for c in cat_features],
    lin_ftrs=LIN_FTRS
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def build_model():
    return BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=1)

def icu24h_loss(preds, targets):
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return nn.BCEWithLogitsLoss()(preds[mask], targets[mask])

# ============================================================
# 5. Train M ensemble members
# ============================================================
print(f"\nTraining Deep Ensemble (M={M}, icu_24h ONLY, mask 포함)...")

ensemble_models = []

for m in range(M):
    print(f"\n{'='*50}")
    print(f"  Member {m+1}/{M}")
    print(f"{'='*50}")

    torch.manual_seed(m * 42)
    np.random.seed(m * 42)

    model     = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auroc_icu24h = 0
    best_epoch             = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for cont, cat, labels in train_loader:
            cont, cat, labels = cont.to(device), cat.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = icu24h_loss(model(static=cont, static_cat=cat)["static"], labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for cont, cat, labels in val_loader:
                cont, cat = cont.to(device), cat.to(device)
                all_preds.append(torch.sigmoid(model(static=cont, static_cat=cat)["static"]).cpu().numpy())
                all_labels.append(labels.numpy())

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        mask = ~np.isnan(all_labels[:, 0])
        if mask.sum() > 0 and len(np.unique(all_labels[mask, 0])) > 1:
            val_auroc_icu24h = roc_auc_score(all_labels[mask, 0], all_preds[mask, 0])
        else:
            val_auroc_icu24h = float('nan')

        print(f"  Epoch {epoch+1:02d}/{EPOCHS} | loss: {train_loss/len(train_loader):.4f} | "
              f"val icu_24h AUROC: {val_auroc_icu24h:.4f}")

        if not np.isnan(val_auroc_icu24h) and val_auroc_icu24h > best_val_auroc_icu24h:
            best_val_auroc_icu24h = val_auroc_icu24h
            best_epoch             = epoch + 1
            # ✅ mask 포함 버전 구분 파일명
            torch.save(model.state_dict(), os.path.join(BASE_DIR, f'ensemble_member_{m}_icu24h_only_mask.pt'))

    print(f"  Best val icu_24h AUROC: {best_val_auroc_icu24h:.4f} at epoch {best_epoch}")
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, f'ensemble_member_{m}_icu24h_only_mask.pt'), map_location=device))
    model.eval()
    ensemble_models.append(model)

print(f"\nAll {M} members trained.")

# ============================================================
# 6. Ensemble prediction
# ============================================================
def ensemble_predict(models, loader, device):
    all_preds_per_model = []
    for model in models:
        model.eval()
        preds = []
        with torch.no_grad():
            for cont, cat, _ in loader:
                cont, cat = cont.to(device), cat.to(device)
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
    entropy   = -(mean_pred * np.log(mean_pred + EPSILON)
                  + (1 - mean_pred) * np.log(1 - mean_pred + EPSILON))

    return mean_pred, variance, entropy, spread, all_labels


print("\nRunning ensemble inference...")
print("  Val set...")
val_mean, val_var, val_entropy, val_spread, val_labels = ensemble_predict(
    ensemble_models, val_loader, device
)
print("  Test set...")
test_mean, test_var, test_entropy, test_spread, test_labels = ensemble_predict(
    ensemble_models, test_loader, device
)
print("  Done.")

mask_val  = ~np.isnan(val_labels[:,  0])
mask_test = ~np.isnan(test_labels[:, 0])

print(f"\n[Sanity] Val  — mean: {val_mean[mask_val,   0].mean():.4f} "
      f"| var: {val_var[mask_val,    0].mean():.6f} "
      f"| entropy: {val_entropy[mask_val, 0].mean():.4f} "
      f"| spread: {val_spread[mask_val, 0].mean():.4f}")
print(f"[Sanity] Test — mean: {test_mean[mask_test, 0].mean():.4f} "
      f"| var: {test_var[mask_test,  0].mean():.6f} "
      f"| entropy: {test_entropy[mask_test, 0].mean():.4f} "
      f"| spread: {test_spread[mask_test, 0].mean():.4f}")

# ============================================================
# 7. Q-model feature extraction
# ============================================================
def extract_qmodel_features(mean_probs, variance, entropy, spread, labels, split_name):
    mask        = ~np.isnan(labels[:, 0])
    prob_icu    = mean_probs[mask, 0]
    var_icu     = variance[mask,   0]
    ent_icu     = entropy[mask,    0]
    spr_icu     = spread[mask,     0]
    true_icu    = labels[mask,     0].astype(int)

    pred_icu    = (prob_icu >= PROB_THRESHOLD).astype(int)
    error_label = (pred_icu != true_icu).astype(int)

    tp = int(((pred_icu == 1) & (true_icu == 1)).sum())
    fp = int(((pred_icu == 1) & (true_icu == 0)).sum())
    fn = int(((pred_icu == 0) & (true_icu == 1)).sum())
    tn = int(((pred_icu == 0) & (true_icu == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    far  = fp / (tp + fp) * 100 if (tp + fp) > 0 else 0

    print(f"\n[{split_name}] threshold={PROB_THRESHOLD}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Sensitivity={sens:.4f} | FAR={far:.2f}% | errors={error_label.sum()}/{len(error_label)}")

    out_df = pd.DataFrame({
        "prob_icu24h": prob_icu, "variance": var_icu, "entropy": ent_icu, "spread": spr_icu,
        "true_label": true_icu, "pred_label": pred_icu, "error_label": error_label,
    })
    fname = os.path.join(CSV_DIR, f"q_features_{split_name}_ensemble_icu24h_only_mask.csv")
    out_df.to_csv(fname, index=False)
    print(f"  Saved → {fname}")
    return out_df

val_features  = extract_qmodel_features(val_mean,  val_var,  val_entropy,  val_spread,  val_labels,  "val")
test_features = extract_qmodel_features(test_mean, test_var, test_entropy, test_spread, test_labels, "test")

print("\n" + "="*60)
print("✅ Deep Ensemble (icu_24h ONLY, mask 포함) pipeline complete.")
print(f"   Models : {BASE_DIR}/ensemble_member_*_icu24h_only_mask.pt")
print("="*60)