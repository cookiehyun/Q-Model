"""
XGBoost base model + Q-model threshold sweep (mask included).

XGBoost now uses the same feature space as the other 3 base models
(raw features + mask columns), so all 4 base models are compared on
equal footing. Everything else follows the same sweep procedure as
the BasicMLP version: base model trained on train/val, Q-model trained
on train, evaluated on test.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Paths / config
# ------------------------------------------------------------
BASE_DIR    = "/user/gaad2403/MDS-ED/key/Final/XGboost"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_DIR     = os.path.join(RESULTS_DIR, "csv")
DATA_PATH   = "/user/gaad2403/MDS-ED/src/data/memmap/mds_ed.csv"
os.makedirs(CSV_DIR, exist_ok=True)

PROB_THRESHOLDS = np.round(np.arange(0.05, 0.21, 0.01), 2)
Q_THRESHOLDS    = np.round(np.arange(0.00, 1.01, 0.01), 2)
ICU24H_IDX      = 1
N_FOLDS         = 5
RANDOM_STATE    = 42
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MLP_HIDDEN     = [64, 32]
MLP_EPOCHS     = 50
MLP_LR         = 1e-3
MLP_BATCH_SIZE = 64
MLP_DROPOUT    = 0.3

XGB_BASE_PARAMS = dict(random_state=RANDOM_STATE, n_jobs=4, eval_metric='logloss')

# ------------------------------------------------------------
# 1. Load & preprocess data (mask included, same feature space
#    as BasicMLP / MC Dropout / Deep Ensemble)
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

demographics_columns = [c for c in df.columns if 'demographics_' in c]
biometrics_columns   = [c for c in df.columns if 'biometrics_' in c]
vitals_columns       = [c for c in df.columns if 'vitals_' in c]
labvalues_columns    = [c for c in df.columns if 'labvalues_' in c]
all_features         = demographics_columns + biometrics_columns + vitals_columns + labvalues_columns

selected_folds = df[df['general_strat_fold'].isin(range(0, 18))]
medians        = selected_folds[all_features].median()

# Add a mask column per feature: 1 if observed, 0 if missing
mask_columns = []
for col in all_features:
    mask_col = col + '_m'
    df[mask_col] = df[col].notna().astype(float)
    mask_columns.append(mask_col)

df[all_features] = df[all_features].fillna(medians)
all_features_with_mask = all_features + mask_columns

target_columns = [
    'deterioration_mortality_1d',
    'deterioration_icu_24h',
    'deterioration_cardiac_arrest',
    'deterioration_vasopressors'
]

# ------------------------------------------------------------
# 2. Train/Val/Test split
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 3. Train XGBoost base model (icu_24h target only)
# ------------------------------------------------------------
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

base_model = XGBClassifier(**XGB_BASE_PARAMS)
base_model.fit(x_tr, y_tr, eval_set=[(x_v, y_v)], verbose=False)

train_prob_icu = base_model.predict_proba(x_tr)[:, 1]  # used to train the Q-model
val_prob_icu   = base_model.predict_proba(x_v)[:, 1]
test_prob_icu  = base_model.predict_proba(x_te)[:, 1]
train_true_icu = y_tr
val_true_icu   = y_v
test_true_icu  = y_te

auroc_val  = roc_auc_score(y_v,  val_prob_icu)
auroc_test = roc_auc_score(y_te, test_prob_icu)
print(f"Val AUROC: {auroc_val:.4f} | Test AUROC: {auroc_test:.4f}")

# ------------------------------------------------------------
# 4. Q-model input features: base features (mask incl.) + base probability
# ------------------------------------------------------------
X_train_features = x_tr
X_test_features  = x_te

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


def get_qprobs(X_train_feat, y_train, X_test_feat, train_prob_icu, test_prob_icu, model_type):
    """Simple (full-train fit) vs CrossFit (5-fold, averaged test predictions)."""
    X_train_q = np.hstack([X_train_feat, train_prob_icu.reshape(-1, 1)]).astype(np.float32)
    X_test_q  = np.hstack([X_test_feat,  test_prob_icu.reshape(-1, 1)]).astype(np.float32)

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
        q_simple, q_cf = get_qprobs(X_train_features, train_err, X_test_features,
                                    train_prob_icu, test_prob_icu, mtype)

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
summary_path = os.path.join(CSV_DIR, "prob_thr_sweep_summary_xgboost_withmask_trainQ.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Summary saved: {summary_path}")
print(f"Base model test AUROC: {auroc_test:.4f}")