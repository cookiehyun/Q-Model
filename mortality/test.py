import pandas as pd
df = pd.read_csv("/user/gaad2403/MDS-ED/key/Final/modality/results/csv/q_features_test_mcdropout_mortality1d_only_mask.csv")
print(df["entropy"].describe())
print(df["entropy"].isna().sum())