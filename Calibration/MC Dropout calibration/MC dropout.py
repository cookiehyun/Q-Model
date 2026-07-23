def mc_dropout_predict(model, test_loader, device, n_samples=50):

    model.train()  # IMPORTANT: keep dropout active

    all_mean_preds = []
    all_uncertainties = []
    all_labels = []

    with torch.no_grad():

        for cont, cat, labels in test_loader:

            cont = cont.to(device)
            cat  = cat.to(device)

            mc_samples = []

            # ----------------------------------------
            # Multiple stochastic forward passes
            # ----------------------------------------
            for _ in range(n_samples):

                out = model(static=cont, static_cat=cat)
                probs = torch.sigmoid(out["static"])

                mc_samples.append(probs.cpu().numpy())

            mc_samples = np.array(mc_samples)

            # -------------
            # 
            # ---------------------------
            # MC statistics
            # ----------------------------------------
            mean_preds = mc_samples.mean(axis=0)
            uncertainty = mc_samples.std(axis=0)

            all_mean_preds.append(mean_preds)
            all_uncertainties.append(uncertainty)
            all_labels.append(labels.numpy())

    all_mean_preds = np.concatenate(all_mean_preds, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_mean_preds, all_uncertainties, all_labels

# =========================================================
# 1. MC Dropout prediction
# =========================================================
print("\nRunning Monte Carlo Dropout...")

all_mean_preds, all_uncertainties, y_true = mc_dropout_predict(
    encoder,
    test_loader,
    device,
    n_samples=50
)

# =========================================================
# 2. Extract ICU_24h only
# =========================================================
target_idx = 0

y_prob = all_mean_preds[:, target_idx]
y_true = y_true[:, target_idx]

mask = ~np.isnan(y_true)

y_prob_mc = y_prob[mask]
y_true_mc = y_true[mask].astype(int)

# =========================================================
# 3. Manual binning (same style as your plot)
# =========================================================
n_bins = 30
bins = np.linspace(0, 1, n_bins + 1)

frac_pos_mc = []
conf = []
bin_sizes = []

ece = 0.0
n = len(y_true_mc)

for i in range(n_bins):

    left, right = bins[i], bins[i + 1]

    if i == n_bins - 1:
        bin_mask = (y_prob_mc >= left) & (y_prob_mc <= right)
    else:
        bin_mask = (y_prob_mc >= left) & (y_prob_mc < right)

    size = np.sum(bin_mask)

    if size == 0:
        frac_pos_mc.append(np.nan)
        conf.append(np.nan)
        bin_sizes.append(0)
        continue

    acc = np.mean(y_true_mc[bin_mask])
    avg_conf = np.mean(y_prob_mc[bin_mask])

    frac_pos_mc.append(acc)
    conf.append(avg_conf)
    bin_sizes.append(size)

    ece += (size / n) * abs(acc - avg_conf)

frac_pos_mc = np.array(frac_pos_mc)
conf = np.array(conf)

centers = (bins[:-1] + bins[1:]) / 2
bin_width = 1.0 / n_bins



# =========================================================
# 5. SAME FOR NEGATIVE CLASS (ICU = 0)
# =========================================================
y_prob_neg_mc = 1 - y_prob_mc
y_true_neg_mc = 1 - y_true_mc

frac_n_mc = []
conf_n = []

ece_neg = 0.0

for i in range(n_bins):

    left, right = bins[i], bins[i + 1]

    if i == n_bins - 1:
        bin_mask = (y_prob_neg_mc >= left) & (y_prob_neg_mc <= right)
    else:
        bin_mask = (y_prob_neg_mc >= left) & (y_prob_neg_mc < right)

    size = np.sum(bin_mask)

    if size == 0:
        frac_n_mc.append(np.nan)
        conf_n.append(np.nan)
        continue

    acc = np.mean(y_true_neg_mc[bin_mask])
    avg_conf = np.mean(y_prob_neg_mc[bin_mask])

    frac_n_mc.append(acc)
    conf_n.append(avg_conf)

    ece_neg += (size / n) * abs(acc - avg_conf)

frac_n_mc = np.array(frac_n_mc)
conf_n = np.array(conf_n)
# =========================================================
# 4-6. Plot positive and negative classes as subplots
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# =========================================================
# LEFT: Positive class (mortality_1y = 1)
# =========================================================

axes[0].plot(
    [0, 1], [0, 1],
    '--',
    color='gray',
    linewidth=1,
    label="Perfect Calibration"
)

axes[0].bar(
    centers,
    frac_pos_mc,
    width=bin_width * 0.9,
    alpha=0.2,
    color='red'
)

axes[0].plot(
    centers,
    frac_pos_mc,
    color='red',
    linewidth=2,
    label="Observed"
)

axes[0].plot(
    centers,
    conf,
    color='black',
    linestyle='--',
    label="Mean predicted prob"
)

axes[0].text(
    0.05,
    0.90,
    f"ECE: {ece:.3f}",
    transform=axes[0].transAxes
)

axes[0].set_xlabel("Predicted probability of mortality_1y")
axes[0].set_ylabel("Fraction of true mortality_1y cases")
axes[0].set_title("Calibration - Positive Class 1 (mortality_1y)")
axes[0].legend(loc='upper right', fontsize=8)
axes[0].grid(alpha=0.3)


# =========================================================
# RIGHT: Negative class (mortality_1y = 0)
# =========================================================

axes[1].plot(
    [0, 1], [0, 1],
    '--',
    color='gray',
    linewidth=1,
    label="Perfect Calibration"
)

axes[1].bar(
    centers,
    frac_n_mc,
    width=bin_width * 0.9,
    alpha=0.2,
    color='blue'
)

axes[1].plot(
    centers,
    frac_n_mc,
    color='blue',
    linewidth=2,
    label="Observed"
)

axes[1].plot(
    centers,
    conf_n,
    color='black',
    linestyle='--',
    label="Mean predicted prob"
)

axes[1].text(
    0.05,
    0.90,
    f"ECE: {ece_neg:.3f}",
    transform=axes[1].transAxes
)

axes[1].set_xlabel("Predicted probability of non-mortality_1y")
axes[1].set_title("Calibration - Negative Class 0 (non-mortality_1y)")
axes[1].legend(loc='upper right',fontsize=8)
axes[1].grid(alpha=0.3)


plt.tight_layout()
plt.show()
