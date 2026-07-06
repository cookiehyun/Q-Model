"""
BasicMLP Base Model — ICU-24h ONLY (단일 타깃 버전)
=====================================================
원본(basicMLP.py, 4개 멀티태스크)에서 icu_24h 단일 타깃으로 변경.
(mcdropout_icu24h_only.py / ensemble_icu24h_only.py와 동일한 방식)

변경 사항:
  - target_dim: 4 → 1 (icu_24h만)
  - label: 4개 타깃 → deterioration_icu_24h만 사용
  - loss: bce_loss_with_nans(4개 합산) → BCEWithLogitsLoss(icu_24h만)
  - val AUROC: icu_24h만 계산 (macro AUROC 없음)
  - 체크포인트 선택: icu_24h AUROC 기준 (원본과 동일하지만 이제 유일한 지표)
  - 저장 파일: best_basicmlp_icu24h_only.pt (원본과 구분)
  - 결과 CSV: q_features_{split}_basicmlp_icu24h_only.csv

나머지 구조(모델 아키텍처, Q-model feature 추출)는 원본과 동일.
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
import warnings
warnings.filterwarnings('ignore')
from clinical_ts.template_modules import ShapeConfig
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import os

# ============================================================
# Model definitions (원본과 동일)
# ============================================================

class BasicEncoderStatic(EncoderStaticBase):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        self.input_channels_cat = hparams_input_shape.static_dim_cat
        self.input_channels_cont = hparams_input_shape.static_dim
        assert(len(hparams_encoder_static.embedding_dims)==hparams_input_shape.static_dim_cat
               and len(hparams_encoder_static.vocab_sizes)==hparams_input_shape.static_dim_cat)
        self.embeddings = nn.ModuleList() if hparams_input_shape.static_dim_cat is not None else None
        for v, e in zip(hparams_encoder_static.vocab_sizes, hparams_encoder_static.embedding_dims):
            self.embeddings.append(nn.Embedding(v, e))
        self.input_dim = int(np.sum(hparams_encoder_static.embedding_dims) + hparams_input_shape.static_dim)
        self.input_channels = hparams_input_shape.static_dim + hparams_input_shape.static_dim_cat

    def embed(self, **kwargs):
        static = kwargs["static"] if "static" in kwargs.keys() else None
        static_cat = kwargs["static_cat"] if "static_cat" in kwargs.keys() else None
        res = []
        if static_cat is not None:
            for i, e in enumerate(self.embeddings):
                res.append(e(static_cat[:, i].long()))
            if static is not None and static_cat is not None:
                res = torch.cat([torch.cat(res, dim=1), static], dim=1)
            else:
                res = torch.cat(res, dim=1)
        else:
            res = static
        return res

    def forward(self, **kwargs):
        raise NotImplementedError

    def get_output_shape(self):
        raise NotImplementedError


@dataclass
class BasicEncoderStaticConfig(EncoderStaticBaseConfig):
    _target_: str = "clinical_ts.tabular.base.BasicEncoderStatic"
    embedding_dims: List[int] = field(default_factory=lambda: [])
    vocab_sizes: List[int]    = field(default_factory=lambda: [])


class BasicEncoderStaticMLP(BasicEncoderStatic):
    def __init__(self, hparams_encoder_static, hparams_input_shape, target_dim=None):
        super().__init__(hparams_encoder_static, hparams_input_shape, target_dim)
        lin_ftrs = [self.input_dim] + list(hparams_encoder_static.lin_ftrs)
        if target_dim is not None and lin_ftrs[-1] != target_dim:
            lin_ftrs.append(target_dim)
        ps = [hparams_encoder_static.dropout] if not isinstance(hparams_encoder_static.dropout, Iterable) else hparams_encoder_static.dropout
        if len(ps) == 1:
            ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
        layers = []
        for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
            layers += bn_drop_lin(ni, no, hparams_encoder_static.batch_norm, p, actn, layer_norm=False)
        self.layers = nn.Sequential(*layers)
        self.output_shape = dataclasses.replace(hparams_input_shape)
        self.output_shape.static_dim = int(lin_ftrs[-1])
        self.output_shape.static_dim_cat = 0

    def forward(self, **kwargs):
        res = self.embed(**kwargs)
        return {"static": self.layers(res)}

    def get_output_shape(self):
        return self.output_shape


@dataclass
class BasicEncoderStaticMLPConfig(BasicEncoderStaticConfig):
    _target_: str = "clinical_ts.tabular.base.BasicEncoderStaticMLP"
    lin_ftrs: List[int] = field(default_factory=lambda: [512])
    dropout: float = 0.5
    batch_norm: bool = True


# ============================================================
# Config
# ============================================================
BATCH_SIZE   = 32
EPOCHS       = 10
LR           = 0.001
WEIGHT_DECAY = 0.001
DROPOUT      = 0.5
LIN_FTRS     = [128, 128, 128]

# ✅ icu_24h ONLY: 단일 타깃 (원본은 4개)
TARGET_TASK = "icu_24h"

# ============================================================
# Directory setup
# ============================================================
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

input_cols = [c for c in df.columns if c.split("_")[0] in ['biometrics', 'demographics', 'labvalues', 'vitals']]

# ✅ [교수님 원안 반영] 딥러닝 모델은 mask 컬럼 포함 (notna 기준: 1=값 있음, 0=결측)
mask_columns = []
for c in input_cols:
    mask_col = c + '_m'
    df[mask_col] = df[c].notna().astype(float)
    mask_columns.append(mask_col)

df_train     = df[df['general_strat_fold'] < 18]
train_medians = df_train[input_cols].median().to_dict()
train_nans   = [c for c, v in df_train[input_cols].isna().sum().to_dict().items() if v > 0]
for c in train_nans:
    df.loc[df[c].isna(), c] = train_medians[c]
df = df.copy()

unique_counts = {c: len(np.unique(np.array(df[c]))) for c in input_cols}
cat_features  = [c for c, v in unique_counts.items()
                 if v < 10 and not c.endswith("nan") and not c.startswith("labvalues")]
cont_features = [c for c in input_cols if c not in cat_features]
# ✅ mask 컬럼은 연속형 취급으로 cont_features에 추가 (embedding 불필요, 0/1 그대로 입력)
cont_features = cont_features + mask_columns

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
# ✅ ethnicity 원-핫 컬럼들이 병합되며 사라지므로, 그에 대응하던 mask 컬럼도 같이 제거
#    (ethnicity는 원래 결측이 없는 데모그래픽 변수라 mask 의미가 없음)
ethnicity_masks = [c + '_m' for c in lbl_itos_ethnicity if (c + '_m') in df.columns]
if ethnicity_masks:
    df.drop(ethnicity_masks, axis=1, inplace=True)
    mask_columns = [c for c in mask_columns if c not in ethnicity_masks]

input_cols    = [c for c in df.columns if c.split("_")[0] in ['biometrics', 'demographics', 'labvalues', 'vitals']]
cat_features  = [c for c in input_cols if c in cat_features]
cont_features = [c for c in input_cols if c not in cat_features]

# ✅ icu_24h만 처리
df["deterioration_" + TARGET_TASK] = df["deterioration_" + TARGET_TASK].replace(-999., np.nan)

print(f"categorical features: {len(cat_features)}, continuous features: {len(cont_features)}")

# ============================================================
# 2. Train / Val / Test split
# ============================================================
train_df = df[df['general_strat_fold'].isin(range(0, 18))].reset_index(drop=True)
val_df   = df[df['general_strat_fold'] == 18].reset_index(drop=True)
test_df  = df[df['general_strat_fold'] == 19].reset_index(drop=True)

val_df  = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)

print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

# ============================================================
# 3. Dataset — ✅ label: icu_24h 단일 컬럼만
# ============================================================
class TabularDataset(Dataset):
    def __init__(self, df, cont_features, cat_features, target_task):
        self.cont   = torch.tensor(df[cont_features].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_features].values,  dtype=torch.long)
        self.labels = torch.tensor(
            df[["deterioration_" + target_task]].values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.cont)

    def __getitem__(self, idx):
        return self.cont[idx], self.cat[idx], self.labels[idx]


train_ds = TabularDataset(train_df, cont_features, cat_features, TARGET_TASK)
val_ds   = TabularDataset(val_df,   cont_features, cat_features, TARGET_TASK)
test_ds  = TabularDataset(test_df,  cont_features, cat_features, TARGET_TASK)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================
# 4. Model
# ============================================================
@dataclass
class MLPConfig:
    embedding_dims: List[int] = field(default_factory=lambda: [16, 16, 16])
    vocab_sizes: List[int]    = field(default_factory=lambda: [2, 5, 5])
    lin_ftrs: List[int]       = field(default_factory=lambda: [128, 128, 128])
    dropout: float = 0.5
    batch_norm: bool = True

@dataclass
class ShapeCfg:
    static_dim: int     = 0
    static_dim_cat: int = 0
    channels: int       = 0
    length: int         = 0
    sequence_last: bool = False
    channels2: int      = 0

shape   = ShapeCfg(static_dim=len(cont_features), static_dim_cat=len(cat_features))
mlp_cfg = MLPConfig(
    embedding_dims=[unique_counts[c] for c in cat_features],
    vocab_sizes=[unique_counts[c] for c in cat_features],
    lin_ftrs=LIN_FTRS
)
# ✅ target_dim=1
encoder = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=1)
print(f"Model parameters: {sum(p.numel() for p in encoder.parameters()):,}")
print(f"Output dim: 1 (icu_24h only)")

# ============================================================
# 5. Loss & optimizer
# ============================================================
def icu24h_loss(preds, targets):
    mask = ~torch.isnan(targets)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    return nn.BCEWithLogitsLoss()(preds[mask], targets[mask])

optimizer = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
encoder   = encoder.to(device)

# ============================================================
# 6. Training loop
#    ✅ 체크포인트 선택: icu_24h AUROC (이제 유일한 지표, macro 없음)
# ============================================================
best_val_auroc_icu24h = 0
best_epoch            = 0

for epoch in range(EPOCHS):
    encoder.train()
    train_loss = 0
    for cont, cat, labels in train_loader:
        cont, cat, labels = cont.to(device), cat.to(device), labels.to(device)
        optimizer.zero_grad()
        out    = encoder(static=cont, static_cat=cat)
        logits = out["static"]
        loss   = icu24h_loss(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    encoder.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for cont, cat, labels in val_loader:
            cont, cat = cont.to(device), cat.to(device)
            out    = encoder(static=cont, static_cat=cat)
            logits = torch.sigmoid(out["static"])
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds,  axis=0)  # (N, 1)
    all_labels = np.concatenate(all_labels, axis=0)  # (N, 1)

    # ✅ icu_24h AUROC만 계산 (macro 없음)
    mask = ~np.isnan(all_labels[:, 0])
    if mask.sum() > 0 and len(np.unique(all_labels[mask, 0])) > 1:
        val_auroc_icu24h = roc_auc_score(all_labels[mask, 0], all_preds[mask, 0])
    else:
        val_auroc_icu24h = float('nan')

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss: {train_loss/len(train_loader):.4f} | "
          f"val icu_24h AUROC: {val_auroc_icu24h:.4f}")

    if not np.isnan(val_auroc_icu24h) and val_auroc_icu24h > best_val_auroc_icu24h:
        best_val_auroc_icu24h = val_auroc_icu24h
        best_epoch            = epoch + 1
        # ✅ 파일명 구분
        torch.save(encoder.state_dict(), os.path.join(BASE_DIR, 'best_basicmlp_icu24h_only.pt'))

print(f"\nBest val icu_24h AUROC: {best_val_auroc_icu24h:.4f} at epoch {best_epoch}")

# ============================================================
# 7. ✅ Inference & Q-model feature extraction
# ============================================================
encoder.load_state_dict(torch.load(os.path.join(BASE_DIR, 'best_basicmlp_icu24h_only.pt'), map_location=device))
encoder.eval()

PROB_THRESHOLD = 0.10   # fixed operating point (sensitivity ≈ 0.80)

def extract_qmodel_features(loader, df_ref, split_name):
    all_probs, all_labels = [], []

    with torch.no_grad():
        for cont, cat, labels in loader:
            cont, cat = cont.to(device), cat.to(device)
            out   = encoder(static=cont, static_cat=cat)
            probs = torch.sigmoid(out["static"])
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)   # (N, 1)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, 1)

    prob_icu  = all_probs[:, 0]
    true_icu  = all_labels[:, 0]

    valid_mask = ~np.isnan(true_icu)

    prob_icu  = prob_icu[valid_mask]
    true_icu  = true_icu[valid_mask]

    pred_icu  = (prob_icu >= PROB_THRESHOLD).astype(int)
    true_icu_int = true_icu.astype(int)

    error_label = (pred_icu != true_icu_int).astype(int)

    tp = int(((pred_icu == 1) & (true_icu_int == 1)).sum())
    fp = int(((pred_icu == 1) & (true_icu_int == 0)).sum())
    fn = int(((pred_icu == 0) & (true_icu_int == 1)).sum())
    tn = int(((pred_icu == 0) & (true_icu_int == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\n[{split_name}] threshold={PROB_THRESHOLD}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Sensitivity={sensitivity:.4f} | error samples={error_label.sum()} / {len(error_label)}")

    out_df = pd.DataFrame({
        "prob_icu24h":  prob_icu,
        "true_label":   true_icu_int,
        "pred_label":   pred_icu,
        "error_label":  error_label,
    })

    # ✅ 파일명 구분
    fname = os.path.join(CSV_DIR, f"q_features_{split_name}_basicmlp_icu24h_only.csv")
    out_df.to_csv(fname, index=False)
    print(f"  Saved → {fname}")
    return out_df


val_features  = extract_qmodel_features(val_loader,  val_df,  "val")
test_features = extract_qmodel_features(test_loader, test_df, "test")

print("\n✅ Done. Q-model input files ready:")
print(f"   {CSV_DIR}/q_features_val_basicmlp_icu24h_only.csv  → use for Q-model training")
print(f"   {CSV_DIR}/q_features_test_basicmlp_icu24h_only.csv → use for Q-model evaluation")