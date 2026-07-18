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


encoder = torch.load(
    "best_realmlp.pt",
    map_location="cpu",
    weights_only=False
)
encoder.eval()

# ----------------------------------------
# Load the trained model
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


encoder = torch.load(
    "best_realmlp.pt",
    map_location=device,
    weights_only=False      # Needed in PyTorch >=2.6
)
encoder.eval()


# ----------------------------------------
# Predict on validation set
# ----------------------------------------
all_preds, all_labels = [], []

with torch.no_grad():

    for cont, cat, labels in val_loader:

        cont = cont.to(DEVICE)
        cat  = cat.to(DEVICE)

        out = encoder(static=cont, static_cat=cat)

        probs = torch.sigmoid(out["static"])

        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

all_preds  = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)


# ----------------------------------------
# Extract mortality_365d predictions and labels
# ----------------------------------------
target_idx = 0

mask = ~np.isnan(all_labels[:, target_idx])

y_true = all_labels[:, target_idx]
y_prob = all_preds[:, target_idx]


# ----------------------------------------
# Save to CSV
# ----------------------------------------
results = pd.DataFrame({
    "Mortality_1y_true": y_true.astype(int),
    "Mortality_1y_probability": y_prob
})


print(results)


# =========================================================
# 1. Load from results table
# =========================================================

mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
y_true = y_true[mask]
y_prob = y_prob[mask]

# =========================================================
# 2. Binning setup
# =========================================================
n_bins = 30
bins = np.linspace(0, 1, n_bins + 1)
centers = (bins[:-1] + bins[1:]) / 2
bin_width = 1.0 / n_bins

# =========================================================
# 3. Helper function (same logic as MC dropout)
# =========================================================
def calibration_stats(y_true, y_prob):
    frac_pos = []
    conf = []
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):

        left, right = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        size = np.sum(mask)

        if size == 0:
            frac_pos.append(np.nan)
            conf.append(np.nan)
            continue

        acc = np.mean(y_true[mask])
        avg_conf = np.mean(y_prob[mask])

        frac_pos.append(acc)
        conf.append(avg_conf)

        ece += (size / n) * abs(acc - avg_conf)

    return np.array(frac_pos), np.array(conf), ece

# =========================================================
# 4. CLASS 1 (ICU = 1)
# =========================================================
frac_pos_1_mlp, conf_1, ece_1 = calibration_stats(y_true, y_prob)

# =========================================================
# 5. CLASS 0 (ICU = 0)
# =========================================================
y_true_0 = 1 - y_true
y_prob_0 = 1 - y_prob

frac_pos_0_mlp, conf_0, ece_0 = calibration_stats(y_true_0, y_prob_0)

# =========================================================
# 6. Plot CLASS 1
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# -------------------------
# CLASS 1 (ICU = 1)
# -------------------------
axes[0].plot([0, 1], [0, 1], '--', color='gray', linewidth=1)

axes[0].bar(
    centers, frac_pos_1_mlp,
    width=bin_width * 0.9,
    alpha=0.2,
    color='red'
)

axes[0].plot(
    centers, frac_pos_1_mlp,
    color='red',
    linewidth=2,
    label="Observed"
)

axes[0].plot(
    centers, conf_1,
    color='black',
    linestyle='--',
    label="Mean predicted prob"
)

axes[0].text(
    0.05, 0.90,
    f"ECE: {ece_1:.3f}",
    transform=axes[0].transAxes
)

axes[0].set_xlabel("Predicted probability of mortality_1y")
axes[0].set_ylabel("Fraction of true mortality_1y cases")
axes[0].set_title("MLP Calibration - Class 1 (mortality_1y = 1)")
axes[0].legend(loc='upper right',fontsize=8)
axes[0].grid(alpha=0.3)


# -------------------------
# CLASS 0 (ICU = 0)
# -------------------------
axes[1].plot([0, 1], [0, 1], '--', color='gray', linewidth=1)

axes[1].bar(
    centers, frac_pos_0_mlp,
    width=bin_width * 0.9,
    alpha=0.2,
    color='blue'
)

axes[1].plot(
    centers, frac_pos_0_mlp,
    color='blue',
    linewidth=2,
    label="Observed"
)

axes[1].plot(
    centers, conf_0,
    color='black',
    linestyle='--',
    label="Mean predicted prob"
)

axes[1].text(
    0.05, 0.90,
    f"ECE: {ece_0:.3f}",
    transform=axes[1].transAxes
)

axes[1].set_xlabel("Predicted probability of non-mortality_1y")
axes[1].set_title("MLP Calibration - Class 0 (mortality_1y = 0)")
axes[1].legend(loc='upper right',fontsize=8)
axes[1].grid(alpha=0.3)


plt.tight_layout()
plt.show()