"""
Q-model threshold sweep for the Deep Ensemble base model.

Same sweep procedure as the BasicMLP version, except the Q-model input
also includes the ensemble's uncertainty statistics (variance, entropy,
spread) in addition to base features + base probability.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
from collections.abc import Iterable
import os
import warnings
warnings.filterwarnings('ignore')

from clinical_ts.template_modules import EncoderStaticBase, EncoderStaticBaseConfig
from clinical_ts.ts.basic_conv1d_modules.basic_conv1d import bn_drop_lin

# ------------------------------------------------------------
# Paths / config
# ------------------------------------------------------------
BASE_DIR    = "/user/gaad2403/MDS-ED/key/Final/DeepEnsemble"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
DATA_PATH   = "/user/gaad2403/MDS-ED/src/data/memmap/mds_ed.csv"
os.makedirs(CSV_DIR, exist_ok=True)

PROB_THRESHOLDS = np.round(np.arange(0.05, 0.21, 0.01), 2)
Q_THRESHOLDS    = np.round(np.arange(0.00, 1.01, 0.01), 2)
ICU24H_IDX      = 0
BATCH_SIZE      = 32
LIN_FTRS        = [128, 128, 128]
M               = 5
EPSILON         = 1e-10
N_FOLDS         = 5
RANDOM_STATE    = 42
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MLP_HIDDEN      = [64, 32]
MLP_EPOCHS      = 50
MLP_LR          = 1e-3
MLP_BATCH_SIZE  = 64
MLP_DROPOUT     = 0.3

# ------------------------------------------------------------
# Model definitions (must match training script)
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

# ------------------------------------------------------------
# 1. Load & preprocess data
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

input_cols    = [c for c in df.columns if c.split("_")[0] in ['biometrics','demographics','labvalues','vitals']]

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

lbl_itos = ["icu_24h"]
for c in lbl_itos:
    df["deterioration_" + c] = df["deterioration_" + c].replace(-999., np.nan)

train_df = df[df['general_strat_fold'].isin(range(0, 18))].reset_index(drop=True)
val_df   = df[df['general_strat_fold'] == 18].reset_index(drop=True)
test_df  = df[df['general_strat_fold'] == 19].reset_index(drop=True)
val_df   = val_df[val_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)
test_df  = test_df[test_df['general_ecg_no_within_stay'] == 0].reset_index(drop=True)

# ------------------------------------------------------------
# 2. Dataset / DataLoader
# ------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, df, cont_f, cat_f, lbl_cols):
        self.cont   = torch.tensor(df[cont_f].values, dtype=torch.float32)
        self.cat    = torch.tensor(df[cat_f].values,  dtype=torch.long)
        self.labels = torch.tensor(df[["deterioration_"+c for c in lbl_cols]].values, dtype=torch.float32)
    def __len__(self): return len(self.cont)
    def __getitem__(self, i): return self.cont[i], self.cat[i], self.labels[i]

train_loader = DataLoader(TabularDataset(train_df, cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
val_loader   = DataLoader(TabularDataset(val_df,  cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(TabularDataset(test_df, cont_features, cat_features, lbl_itos),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ------------------------------------------------------------
# 3. Load ensemble members, run inference
# ------------------------------------------------------------
shape   = ShapeCfg(static_dim=len(cont_features), static_dim_cat=len(cat_features))
mlp_cfg = MLPConfig(
    embedding_dims=[unique_counts[c] for c in cat_features],
    vocab_sizes=[unique_counts[c] for c in cat_features],
    lin_ftrs=LIN_FTRS
)

ensemble_models = []
for m in range(M):
    pt_path = os.path.join(BASE_DIR, f"ensemble_member_{m}_icu24h_only_mask.pt")
    model   = BasicEncoderStaticMLP(mlp_cfg, shape, target_dim=len(lbl_itos)).to(DEVICE)
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    model.eval()
    ensemble_models.append(model)


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


train_mean, train_var, train_ent, train_spr, train_labels = ensemble_predict(ensemble_models, train_loader)
val_mean,  val_var,  val_ent,  val_spr,  val_labels  = ensemble_predict(ensemble_models, val_loader)
test_mean, test_var, test_ent, test_spr, test_labels = ensemble_predict(ensemble_models, test_loader)

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

# ------------------------------------------------------------
# 4. Q-model input features: base features + ensemble stats
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 5. Q-model helpers
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
        m = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_eval)[:, 1]
    elif model_type == "MLP":
        return train_mlp(X_tr, y_tr.astype(np.float32), X_eval)
    else:
        m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           eval_metric='logloss',
                           random_state=RANDOM_STATE, verbosity=0)
        m.fit(X_tr, y_tr)
        return m.predict_proba(X_eval)[:, 1]


def get_qprobs(X_train_feat, y_train, X_test_feat,
               train_prob_icu, train_var_icu, train_ent_icu, train_spr_icu,
               test_prob_icu, test_var_icu, test_ent_icu, test_spr_icu,
               model_type):
    """Simple (full-train fit) vs CrossFit (5-fold, averaged test predictions)."""
    X_train_q = np.hstack([
        X_train_feat, train_prob_icu.reshape(-1,1), train_var_icu.reshape(-1,1),
        train_ent_icu.reshape(-1,1), train_spr_icu.reshape(-1,1)
    ]).astype(np.float32)
    X_test_q = np.hstack([
        X_test_feat, test_prob_icu.reshape(-1,1), test_var_icu.reshape(-1,1),
        test_ent_icu.reshape(-1,1), test_spr_icu.reshape(-1,1)
    ]).astype(np.float32)

    simple = fit_predict(model_type, X_train_q, y_train, X_test_q)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    test_fold_preds = []
    for tri, _ in skf.split(X_train_q, y_train):
        X_tr, y_tr = X_train_q[tri], y_train[tri]
        test_fold_preds.append(fit_predict(model_type, X_tr, y_tr, X_test_q))
    cf = np.mean(test_fold_preds, axis=0)

    return simple, cf


def best_op(q_probs, pred_base, true_label, fp_base):
    best = None
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
        if sens >= 0.80 and (best is None or fpr > best["FP_reduction_pct"]):
            best = dict(q_thr=q_thr, sensitivity=round(sens,4), specificity=round(spec,4),
                        TP=tp, FP=fp, FN=fn, TN=tn, FP_reduction_pct=round(fpr,2))
    return best


# ------------------------------------------------------------
# 6. Main sweep
# ------------------------------------------------------------
summary_rows = []

for prob_thr in PROB_THRESHOLDS:
    print(f">>> prob_thr = {prob_thr:.2f}")

    train_pred = (train_prob_icu >= prob_thr).astype(int)
    test_pred  = (test_prob_icu  >= prob_thr).astype(int)
    train_err  = (train_pred != train_true_icu).astype(int)
    test_err   = (test_pred  != test_true_icu).astype(int)

    tp_b = int(((test_pred==1)&(test_true_icu==1)).sum())
    fp_b = int(((test_pred==1)&(test_true_icu==0)).sum())
    fn_b = int(((test_pred==0)&(test_true_icu==1)).sum())
    tn_b = int(((test_pred==0)&(test_true_icu==0)).sum())
    sens_b = tp_b/(tp_b+fn_b) if (tp_b+fn_b)>0 else 0

    summary_rows.append(dict(
        prob_thr=prob_thr, strategy="Baseline", model="-",
        base_sensitivity=round(sens_b,4), best_q_thr="-", sensitivity=round(sens_b,4),
        FP_reduction_pct=0.0, TP=tp_b, FP=fp_b, FN=fn_b, TN=tn_b
    ))

    for mtype in ["LR", "MLP", "XGB"]:
        q_simple, q_cf = get_qprobs(
            X_train_features, train_err, X_test_features,
            train_prob_icu, train_var_icu, train_ent_icu, train_spr_icu,
            test_prob_icu, test_var_icu, test_ent_icu, test_spr_icu,
            mtype
        )

        for strategy, q_probs in [("Simple", q_simple), ("CrossFit", q_cf)]:
            best = best_op(q_probs, test_pred, test_true_icu, fp_b)
            if best:
                summary_rows.append(dict(
                    prob_thr=prob_thr, strategy=strategy, model=mtype,
                    base_sensitivity=round(sens_b,4), best_q_thr=best["q_thr"],
                    sensitivity=best["sensitivity"], FP_reduction_pct=best["FP_reduction_pct"],
                    TP=best["TP"], FP=best["FP"], FN=best["FN"], TN=best["TN"]
                ))
            else:
                summary_rows.append(dict(
                    prob_thr=prob_thr, strategy=strategy, model=mtype,
                    base_sensitivity=round(sens_b,4), best_q_thr="-", sensitivity="<0.80",
                    FP_reduction_pct="-", TP="-", FP="-", FN="-", TN="-"
                ))

# ------------------------------------------------------------
# 7. Save summary
# ------------------------------------------------------------
summary_df   = pd.DataFrame(summary_rows)
summary_path = os.path.join(CSV_DIR, "prob_thr_sweep_summary_ensemble_icu24h_only_mask_trainQ.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved: {summary_path}")