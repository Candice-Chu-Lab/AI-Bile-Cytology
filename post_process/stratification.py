import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from stratification_postprocess import create_master_summary
import os
from pathlib import Path




# create a cross-validation folder if it doesn't exist
df = pd.read_excel("./Annotation_Positive_Selected/total_count_original.xlsx")
os.makedirs("cross-validation", exist_ok=True)

# We want to manually delete two cases
excluded_cases = ["DS_A09R_07S", "DS_A09R_18S", "DS_A09R_08S", "DS_A01R_17S"]
df = df[~df["slide_name"].isin(excluded_cases)].reset_index(drop=True)
df = df.drop(columns=["Not-usable"])


X = df["slide_name"].values
y = df[["Present", "Not-present"]].values
y_alter = df[
    [
        "Present",
        "Not-present",
        "Rods",
        "Cocci",
        "Yeast",
        "Few",
        "Moderate",
        "Abundant",
    ]
].to_numpy()

# Step 1: Split off 20% as held-out test set (1 fold of 5 = 20%)
outer = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dev_idx, test_idx = next(outer.split(X, y_alter))  # take just the first split

dev_df  = df.iloc[dev_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

test_df.to_csv("cross-validation/test.csv", index=False)
print(f"Test  => cases: {len(test_df)}, pos: {test_df['Present'].sum()}, neg: {test_df['Not-present'].sum()}")



# Step 2: 3-fold CV on the remaining 80%
X_dev = dev_df["slide_name"].values
y_dev = dev_df[["Present", "Not-present"]].values
y_alter_dev = dev_df[
    [
        "Present",
        "Not-present",
        "Rods",
        "Cocci",
        "Yeast",
        "Few",
        "Moderate",
        "Abundant",
    ]
].values


for state_num in range(41, 43, 1):
    inner = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=state_num)

    for i, (train_idx, val_idx) in enumerate(inner.split(X_dev, y_alter_dev)):
        train_df = dev_df.iloc[train_idx]
        val_df   = dev_df.iloc[val_idx]

        # stop at the first available split

        os.makedirs(f"cross-validation/state_{state_num}", exist_ok=True)
        train_df.to_csv(f"cross-validation/state_{state_num}/fold_{i}_train.csv", index=False)
        val_df.to_csv(f"cross-validation/state_{state_num}/fold_{i}_val.csv", index=False)
        print(f"Fold {i}: train => cases: {len(train_df)}, pos: {train_df['Present'].sum()} | "
            f"val => cases: {len(val_df)}, pos: {val_df['Present'].sum()}")

    fit_scenario = create_master_summary(Path(f"cross-validation/state_{state_num}/"))
    if fit_scenario:
        print(f"State {state_num} is a valid split scenario.")
        break





# def validation_fold_is_valid(
#     val_df: pd.DataFrame,
#     minimum_count: int = 1,
# ) -> bool:
#     """
#     Return True only when every required label appears at least
#     minimum_count times in the validation fold.
#     """
#     return all(
#         val_df[column].sum() >= minimum_count
#         for column in REQUIRED_VAL_LABELS
#     )


# accepted_seed = None
# accepted_splits = None

# for seed in range(1000):
#     splitter = MultilabelStratifiedKFold(
#         n_splits=3,
#         shuffle=True,
#         random_state=seed,
#     )

#     candidate_splits = []
#     all_folds_valid = True

#     for fold, (train_idx, val_idx) in enumerate(
#         splitter.split(X_dev, y_alter_dev)
#     ):
#         train_df = dev_df.iloc[train_idx].copy()
#         val_df = dev_df.iloc[val_idx].copy()

#         if not validation_fold_is_valid(val_df):
#             all_folds_valid = False
#             break

#         candidate_splits.append(
#             {
#                 "fold": fold,
#                 "train_df": train_df,
#                 "val_df": val_df,
#             }
#         )

#     # Accept the seed only if all three validation folds are valid.
#     if all_folds_valid and len(candidate_splits) == 3:
#         accepted_seed = seed
#         accepted_splits = candidate_splits
#         break


# if accepted_seed is None:
#     raise RuntimeError(
#         "No acceptable three-fold split was found within 1000 seeds."
#     )


# state_folder = OUTPUT_ROOT / f"state_{accepted_seed}"