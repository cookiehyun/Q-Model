"""
prob_threshold sweep experiment for BasicMLP Q-model — icu_24h ONLY
======================================================================
변경 사항 (교수님 지침 반영):
  1. Mask 컬럼 추가: 딥러닝 base 모델(BasicMLP)은 원본 + mask 피처 사용
     (교수님 demo.ipynb: "use all_features_with_mask for deep learning")
     mask는 notna() 기준 (1=값 있음, 0=결측) — base 모델과 동일 컬럼셋이어야
     체크포인트(best_basicmlp_icu24h_only.pt)의 input_dim과 일치함.
  2. Q-model 학습 데이터: val -> train으로 변경
     (uncertainty/feature가 train 분포에서 온 것이므로 train 사용)
     - Q-model 학습(Simple/CrossFit fit): train set
     - Q-model 평가 & 최종 baseline 지표: test set (변경 없음)

주의: train set은 val보다 훨씬 크므로(약 18배) LR/MLP/XGB 학습 및
   5-fold CrossFit 학습 시간이 크게 늘어납니다.
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
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler   # ← 이 줄 추가
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
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
BASE_DIR    = "/fs/dss/home/gaad2403/MDS-ED/key/Final/basicMLP"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
PNG_DIR     = os.path.join(RESULTS_DIR, "png")
DATA_PATH   = "/user/gaad2403/MDS-ED/src/data/memmap/mds_ed.csv"
PT_PATH     = os.path.join(BASE_DIR, "best_basicmlp_icu24h_only.pt")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)

PROB_THRESHOLDS = np.round(np.arange(0.05, 0.21, 0.01), 2)
Q_THRESHOLDS    = np.round(np.arange(0.00, 1.01, 0.01), 2)
ICU24H_IDX      = 0
BATCH_SIZE      = 32
LIN_FTRS        = [128, 128, 128]
N_FOLDS         = 5
RANDOM_STATE    = 42
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MLP_HIDDEN     = [64, 32]
MLP_EPOCHS     = 50
MLP_LR         = 1e-3
MLP_BATCH_SIZE = 512
MLP_DROPOUT    = 0.3

print(f"prob_threshold sweep: {PROB_THRESHOLDS}")
print(f"Device: {DEVICE}")

# ============================================================
# Model definitions (원본과 동일)
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
        ps = [hparams_encoder_static.dropout] if not isinstance(hparams_encoder_static.dropout, Iterable) else hparams_encoder_static.dropout
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
# 1. Load & preprocess data
# ============================================================
print("\nLoading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

input_cols = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]

# ✅ [교수님 원안] 딥러닝 base 모델과 동일하게 mask 컬럼 추가 (notna 기준)
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
cont_features = cont_features + mask_columns  # ✅ mask 추가

df["vitals_acuity"] = df["vitals_acuity"].apply(lambda x: int(x) - 1)
lbl_eth = ['demographics_ethnicity_asian','demographics_ethnicity_black/african',
           'demographics_ethnicity_hispanic/latino','demographics_ethnicity_other',
           'demographics_ethnicity_white']
df["demographics_ethnicity"] = df.apply(lambda r: np.where([r[c] for c in lbl_eth])[0][0], axis=1)
df.drop(lbl_eth, axis=1, inplace=True)
# ✅ ethnicity 원-핫이 병합되며 사라지므로 대응 mask도 제거
ethnicity_masks = [c + '_m' for c in lbl_eth if (c + '_m') in df.columns]
if ethnicity_masks:
    df.drop(ethnicity_masks, axis=1, inplace=True)
    mask_columns = [c for c in mask_columns if c not in ethnicity_masks]

input_cols    = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]
cat_features  = [c for c in input_cols if c in cat_features]
cont_features = [c for c in input_cols if c not in cat_features]  # mask 자동 재포함됨

lbl_itos = ["icu_24h"]
for c in lbl_itos:
    df["deterioration_" + c] = df["deterioration_" + c].replace(-999., np.nan)

# ✅ train_df 추가 (Q-model 학습용)
train_df = df[df['general_strat_fold'].isin(range(0, 18))].reset_index(drop=True)
val_df   = df[df['general_strat_fold'] == 18].reset_index(drop=True)
test_df  = df[df['general_strat_fold'] == 19].reset_index(drop=True)
val_df   = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df  = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
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

# ✅ train_loader 추가
train_loader = DataLoader(TabularDataset(train_df, cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(TabularDataset(val_df,  cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TabularDataset(test_df, cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 3. Load pretrained base model
# ============================================================
shape   = ShapeCfg(static_dim=len(cont_features), static_dim_cat=len(cat_features))
mlp_cfg = MLPConfig(
    embedding_dims=[unique_counts[c] for c in cat_features],
    vocab_sizes=[unique_counts[c] for c in cat_features],
    lin_ftrs=LIN_FTRS
)
encoder = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=1).to(DEVICE)
encoder.load_state_dict(torch.load(PT_PATH, map_location=DEVICE))
encoder.eval()
print(f"Loaded: {PT_PATH}")

def get_probs_labels(loader):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for cont, cat, labels in loader:
            cont, cat = cont.to(DEVICE), cat.to(DEVICE)
            probs = torch.sigmoid(encoder(static=cont, static_cat=cat)["static"])
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs, 0), np.concatenate(all_labels, 0)

print("Extracting probabilities...")
train_probs, train_labels = get_probs_labels(train_loader)   # ✅ train도 추론
val_probs,   val_labels   = get_probs_labels(val_loader)
test_probs,  test_labels  = get_probs_labels(test_loader)

train_prob_icu = train_probs[:, ICU24H_IDX]
train_true_icu = train_labels[:, ICU24H_IDX]
train_mask     = ~np.isnan(train_true_icu)
train_prob_icu = train_prob_icu[train_mask]
train_true_icu = train_true_icu[train_mask].astype(int)

val_prob_icu   = val_probs[:, ICU24H_IDX]
val_true_icu   = val_labels[:, ICU24H_IDX]
val_mask       = ~np.isnan(val_true_icu)
val_prob_icu   = val_prob_icu[val_mask]
val_true_icu   = val_true_icu[val_mask].astype(int)

test_prob_icu  = test_probs[:, ICU24H_IDX]
test_true_icu  = test_labels[:, ICU24H_IDX]
test_mask      = ~np.isnan(test_true_icu)
test_prob_icu  = test_prob_icu[test_mask]
test_true_icu  = test_true_icu[test_mask].astype(int)

print(f"Train ICU samples: {len(train_prob_icu)}")
print(f"Val   ICU samples: {len(val_prob_icu)}")
print(f"Test  ICU samples: {len(test_prob_icu)}")

# ============================================================
# ✅ Prepare Q-Model features: Base features (mask 포함) + Base probability
#    ✅ train 기준으로 준비 (Q-model 학습용)
# ============================================================
print("\n" + "="*70)
print("Preparing Q-Model features (TRAIN set, base features + base probability)...")
print("="*70)

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

print(f"\n📊 Q-Model Input Features:")
print(f"  Continuous (mask 포함): {len(cont_features)}, Categorical: {len(cat_features)}")
print(f"  Total base: {X_train_features.shape[1]}, + Base prob: 1")
print(f"  = Total Q-Model features: {X_train_features.shape[1] + 1}")

feature_names_qmodel = cont_features + cat_features + ["base_model_prob_icu24h"]

# ============================================================
# 4. Q-model helpers (unchanged)
# ============================================================
class QModelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_eval):
    model   = QModelMLP(X_tr.shape[1], MLP_HIDDEN, MLP_DROPOUT).to(DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                  torch.tensor(y_tr, dtype=torch.float32)),
                    batch_size=MLP_BATCH_SIZE, shuffle=True)
    for _ in range(MLP_EPOCHS):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss_fn(torch.sigmoid(model(xb)), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X_eval, dtype=torch.float32).to(DEVICE))).cpu().numpy()


def fit_predict(model_type, X_tr, y_tr, X_eval):
    if model_type == "LR":
        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_eval_s = scaler.transform(X_eval)
        m = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, C=1e6)
        m.fit(X_tr_s, y_tr)
        return m.predict_proba(X_eval_s)[:, 1]
    elif model_type == "MLP":
        return train_mlp(X_tr, y_tr.astype(np.float32), X_eval)
    else:  # XGB
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           tree_method='hist', device='cuda',
                           eval_metric='logloss',
                           random_state=RANDOM_STATE, verbosity=0)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_eval)[:, 1]
    
# ✅ 함수 인자명은 val 그대로 두되(호환성), 실제로는 train 데이터를 넘겨서 사용
def get_qprobs(X_val_feat, y_val, X_test_feat, val_prob_icu, test_prob_icu,
               model_type, verbose_label=""):
    X_val_q  = np.hstack([X_val_feat, val_prob_icu.reshape(-1, 1)]).astype(np.float32)
    X_test_q = np.hstack([X_test_feat, test_prob_icu.reshape(-1, 1)]).astype(np.float32)

    simple = fit_predict(model_type, X_val_q, y_val, X_test_q)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_probs       = np.zeros(len(X_val_q))
    test_fold_preds = []

    for tri, vli in skf.split(X_val_q, y_val):
        X_tr, y_tr = X_val_q[tri], y_val[tri]
        X_vl       = X_val_q[vli]
        fold_oof_pred  = fit_predict(model_type, X_tr, y_tr, X_vl)
        fold_test_pred = fit_predict(model_type, X_tr, y_tr, X_test_q)
        oof_probs[vli] = fold_oof_pred
        test_fold_preds.append(fold_test_pred)

    cf = np.mean(test_fold_preds, axis=0)

    if len(np.unique(y_val)) > 1:
        oof_auroc = roc_auc_score(y_val, oof_probs)
        oof_brier = brier_score_loss(y_val, oof_probs)
        print(f"    [{model_type} CF{verbose_label}] OOF AUROC={oof_auroc:.4f} | OOF Brier={oof_brier:.4f}")

    return simple, cf


def best_operating_point(q_probs, pred_base, true_label, fp_base, tp_base, fn_base):
    best_row = None
    for q_thr in Q_THRESHOLDS:
        pred_new = pred_base.copy()
        pred_new[q_probs >= q_thr] = 0
        tp = int(((pred_new==1)&(true_label==1)).sum())
        fp = int(((pred_new==1)&(true_label==0)).sum())
        fn = int(((pred_new==0)&(true_label==1)).sum())
        tn = int(((pred_new==0)&(true_label==0)).sum())
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        fpr  = (fp_base-fp)/fp_base*100 if fp_base>0 else 0
        if sens >= 0.80:
            if best_row is None or fpr > best_row["FP_reduction_pct"]:
                best_row = dict(q_thr=q_thr, sensitivity=round(sens,4),
                                specificity=round(spec,4),
                                TP=tp, FP=fp, FN=fn, TN=tn,
                                FP_reduction_pct=round(fpr,2))
    return best_row


def full_sweep(q_probs, pred_base, true_label, fp_base):
    rows = []
    for q_thr in Q_THRESHOLDS:
        pred_new = pred_base.copy()
        pred_new[q_probs >= q_thr] = 0
        tp = int(((pred_new==1)&(true_label==1)).sum())
        fp = int(((pred_new==1)&(true_label==0)).sum())
        fn = int(((pred_new==0)&(true_label==1)).sum())
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        fpr  = (fp_base-fp)/fp_base*100 if fp_base>0 else 0
        rows.append(dict(q_thr=q_thr, sensitivity=sens, FP_reduction_pct=fpr))
    return pd.DataFrame(rows)

# ============================================================
# 5. Main sweep loop
#    ✅ Q-model 학습: train_prob_icu / train_true_icu / X_train_features
#    baseline & 최종 평가: test set (변경 없음)
# ============================================================
summary_rows = []
sweep_data   = {}

print(f"\n{'='*60}")
print(f"Starting prob_threshold sweep: {PROB_THRESHOLDS}")
print(f"{'='*60}")

for prob_thr in PROB_THRESHOLDS:
    print(f"\n>>> prob_threshold = {prob_thr:.2f}")

    train_pred = (train_prob_icu >= prob_thr).astype(int)   # ✅ train 기준
    test_pred  = (test_prob_icu  >= prob_thr).astype(int)
    train_err  = (train_pred != train_true_icu).astype(int)  # ✅ Q-model 타깃 (train)
    test_err   = (test_pred  != test_true_icu).astype(int)

    tp_b = int(((test_pred==1)&(test_true_icu==1)).sum())
    fp_b = int(((test_pred==1)&(test_true_icu==0)).sum())
    fn_b = int(((test_pred==0)&(test_true_icu==1)).sum())
    tn_b = int(((test_pred==0)&(test_true_icu==0)).sum())
    sens_b = tp_b/(tp_b+fn_b) if (tp_b+fn_b)>0 else 0

    print(f"  Baseline(test): TP={tp_b} FP={fp_b} FN={fn_b} TN={tn_b} | Sens={sens_b:.4f}")
    print(f"  Train errors: {train_err.sum()} / {len(train_err)}")

    summary_rows.append(dict(
        prob_thr=prob_thr, strategy="Baseline", model="—",
        base_sensitivity=round(sens_b,4),
        best_q_thr="—", sensitivity=round(sens_b,4),
        FP_reduction_pct=0.0,
        TP=tp_b, FP=fp_b, FN=fn_b, TN=tn_b,
        AUROC="—", Brier="—"
    ))

    for mtype in ["LR", "MLP", "XGB"]:
        # ✅ train 데이터로 Q-model 학습, test로 평가
        q_simple, q_cf = get_qprobs(X_train_features, train_err, X_test_features,
                                    train_prob_icu, test_prob_icu,
                                    mtype,
                                    verbose_label=f" @thr={prob_thr:.2f}")

        for strategy, q_probs in [("Simple", q_simple), ("CrossFit", q_cf)]:
            auroc = roc_auc_score(test_err, q_probs) if len(np.unique(test_err))>1 else float('nan')
            brier = brier_score_loss(test_err, q_probs)

            best = best_operating_point(q_probs, test_pred, test_true_icu, fp_b, tp_b, fn_b)
            sweep_data[(prob_thr, mtype, strategy)] = full_sweep(q_probs, test_pred, test_true_icu, fp_b)

            if best:
                summary_rows.append(dict(
                    prob_thr=prob_thr, strategy=strategy, model=mtype,
                    base_sensitivity=round(sens_b,4),
                    best_q_thr=best["q_thr"],
                    sensitivity=best["sensitivity"],
                    FP_reduction_pct=best["FP_reduction_pct"],
                    TP=best["TP"], FP=best["FP"], FN=best["FN"], TN=best["TN"],
                    AUROC=round(auroc,4), Brier=round(brier,4)
                ))
            else:
                summary_rows.append(dict(
                    prob_thr=prob_thr, strategy=strategy, model=mtype,
                    base_sensitivity=round(sens_b,4),
                    best_q_thr="—", sensitivity="<0.80",
                    FP_reduction_pct="—",
                    TP="—", FP="—", FN="—", TN="—",
                    AUROC=round(auroc,4), Brier=round(brier,4)
                ))

        print(f"  {mtype} done")

# ============================================================
# 6. Save summary CSV
# ============================================================
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(CSV_DIR, "prob_thr_sweep_summary_icu24h_only_mask_trainQ.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary saved: {summary_path}")
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("✅ All done! (mask 포함, Q-model train set 기준)")
print("="*70)
print(f"  Summary CSV: {summary_path}")
print("="*70)