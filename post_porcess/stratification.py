import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

df = pd.read_csv("annotation_result.csv")
X = df["slide_name"].values
y = df[["positive", "negative"]].values

# Step 1: Split off 20% as held-out test set (1 fold of 5 = 20%)
outer = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dev_idx, test_idx = next(outer.split(X, y))  # take just the first split

dev_df  = df.iloc[dev_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

test_df.to_csv("cross-validation/test.csv", index=False)
print(f"Test  => cases: {len(test_df)}, pos: {test_df['positive'].sum()}, neg: {test_df['negative'].sum()}")

# Step 2: 3-fold CV on the remaining 80%
X_dev = dev_df["slide_name"].values
y_dev = dev_df[["positive", "negative"]].values

inner = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for i, (train_idx, val_idx) in enumerate(inner.split(X_dev, y_dev)):
    train_df = dev_df.iloc[train_idx]
    val_df   = dev_df.iloc[val_idx]
    train_df.to_csv(f"cross-validation/fold_{i}_train.csv", index=False)
    val_df.to_csv(f"cross-validation/fold_{i}_val.csv", index=False)
    print(f"Fold {i}: train => cases: {len(train_df)}, pos: {train_df['positive'].sum()} | "
          f"val => cases: {len(val_df)}, pos: {val_df['positive'].sum()}")