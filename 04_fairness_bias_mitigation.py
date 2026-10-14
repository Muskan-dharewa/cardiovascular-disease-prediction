
from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.sampling import Condition

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
TEST_SIZE = 0.20
CTGAN_EPOCHS = 300

INPUT_PATH = Path("data/combined_statslogExcluded_SMOTE_SexBinary.csv")
BALANCED_PATH = Path(
    "data/combined_statslogExcluded_GenderBalancedWithCTGAN.csv"
)
TARGET = "target"
SENSITIVE = "sex"
CLINICAL_DROP_COLS = ["chol", "restecg", "slope", "oldpeak"]

random.seed(SEED)
np.random.seed(SEED)
def calculate_fairness(y_true, y_pred, sex):
    return {
        "DPD": demographic_parity_difference(
            y_true,
            y_pred,
            sensitive_features=sex,
        ),
        "DPR": demographic_parity_ratio(
            y_true,
            y_pred,
            sensitive_features=sex,
        ),
        "EOD": equalized_odds_difference(
            y_true,
            y_pred,
            sensitive_features=sex,
        ),
        "EOR": equalized_odds_ratio(
            y_true,
            y_pred,
            sensitive_features=sex,
        ),
    }


# ============================================================
# 3. MODEL DEFINITIONS
# ============================================================

def make_model(model_name):
    if model_name == "Clinical XGBoost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=153,
            max_depth=3,
            learning_rate=0.2017,
            subsample=0.94,
            colsample_bytree=0.9488,
            gamma=3.5160,
            reg_alpha=0.7811,
            reg_lambda=0.6871,
            random_state=SEED,
        )

    return RandomForestClassifier(
        n_estimators=110,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=8,
        max_features="sqrt",
        bootstrap=True,
        random_state=SEED,
    )


def get_features(data, model_name):
    X = data.drop(columns=[TARGET]).copy()

    if model_name == "Non-Clinical Random Forest":
        X = X.drop(
            columns=CLINICAL_DROP_COLS,
            errors="ignore",
        )

    return X


# ============================================================
# 4. LOAD AND VALIDATE ORIGINAL DATA
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

df = pd.read_csv(INPUT_PATH)

required = {TARGET, SENSITIVE}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df[[TARGET, SENSITIVE]].isna().any().any():
    raise ValueError("Missing values in sex or target.")

if set(df[TARGET].unique()) != {0, 1}:
    raise ValueError("Target must contain only 0 and 1.")

if set(df[SENSITIVE].unique()) != {0, 1}:
    raise ValueError("Sex must contain only 0 and 1.")

print("=" * 70)
print("CTGAN GENDER-BIAS MITIGATION AND FAIRNESS EVALUATION")
print("=" * 70)

print("\nOriginal dataset shape:", df.shape)
print("\nOriginal sex-target counts:")
print(pd.crosstab(df[SENSITIVE], df[TARGET]))


# ============================================================
# 5. CREATE A HELD-OUT TEST SET FIRST
# ============================================================
# CTGAN must not see the test records.
# Both before/after models will be evaluated on this same test set.

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df[TARGET],
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nOriginal training rows:", len(train_df))
print("Held-out test rows:", len(test_df))

print("\nTraining sex-target counts:")
print(pd.crosstab(train_df[SENSITIVE], train_df[TARGET]))


# ============================================================
# 6. DETERMINE TRAINING GROUP COUNTS
# ============================================================

group_counts = train_df.groupby([SENSITIVE, TARGET]).size()

desired_per_target = {
    target_value: max(
        int(group_counts.get((sex_value, target_value), 0))
        for sex_value in [0, 1]
    )
    for target_value in [0, 1]
}

print("\nDesired training records per sex within each target:")
for target_value, count in desired_per_target.items():
    label = "Healthy" if target_value == 0 else "CVD"
    print(f"{label}: {count} per sex group")


# ============================================================
# 7. TRAIN CTGAN ON TRAINING DATA ONLY
# ============================================================

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=train_df)

metadata.update_column(
    column_name=SENSITIVE,
    sdtype="categorical",
)
metadata.update_column(
    column_name=TARGET,
    sdtype="categorical",
)

print("\nTraining CTGAN...")

synthesizer = CTGANSynthesizer(
    metadata=metadata,
    epochs=CTGAN_EPOCHS,
    verbose=True,
    enforce_min_max_values=True,
    enforce_rounding=True,
)

synthesizer.fit(train_df)

print("CTGAN training completed.")


# ============================================================
# 8. GENERATE RECORDS FOR UNDERREPRESENTED GROUPS
# ============================================================

conditions = []

for sex_value in [0, 1]:
    for target_value in [0, 1]:
        current_count = int(
            group_counts.get((sex_value, target_value), 0)
        )

        needed = desired_per_target[target_value] - current_count

        if needed > 0:
            conditions.append(
                Condition(
                    num_rows=needed,
                    column_values={
                        SENSITIVE: sex_value,
                        TARGET: target_value,
                    },
                )
            )

if conditions:
    print("\nGenerating synthetic training records...")

    synthetic_df = synthesizer.sample_from_conditions(
        conditions=conditions,
    )
else:
    synthetic_df = pd.DataFrame(columns=train_df.columns)

synthetic_df = synthetic_df[train_df.columns]

if not synthetic_df.empty:
    if not synthetic_df[SENSITIVE].isin([0, 1]).all():
        raise ValueError("Invalid synthetic sex category.")

    if not synthetic_df[TARGET].isin([0, 1]).all():
        raise ValueError("Invalid synthetic target category.")

balanced_train = pd.concat(
    [train_df, synthetic_df],
    ignore_index=True,
)

final_counts = balanced_train.groupby([SENSITIVE, TARGET]).size()

print("\nSynthetic records generated:", len(synthetic_df))
print("Balanced training shape:", balanced_train.shape)

print("\nBalanced training sex-target counts:")
print(pd.crosstab(balanced_train[SENSITIVE], balanced_train[TARGET]))

for target_value in [0, 1]:
    count_0 = int(final_counts.get((0, target_value), 0))
    count_1 = int(final_counts.get((1, target_value), 0))

    if count_0 != count_1:
        raise ValueError(
            f"Balancing failed for target={target_value}: "
            f"sex 0={count_0}, sex 1={count_1}"
        )

BALANCED_PATH.parent.mkdir(parents=True, exist_ok=True)
balanced_train.to_csv(BALANCED_PATH, index=False)

print("\nTraining-set gender balance: PASSED")
print("Saved balanced training data:", BALANCED_PATH)


# ============================================================
# 9. EVALUATE BEFORE AND AFTER CTGAN
# ============================================================

results = []

models = [
    "Clinical XGBoost",
    "Non-Clinical Random Forest",
]

training_sets = [
    ("Before CTGAN", train_df),
    ("After CTGAN", balanced_train),
]

for model_name in models:
    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    X_test = get_features(test_df, model_name)
    y_test = test_df[TARGET]
    sex_test = test_df[SENSITIVE]

    for stage, training_data in training_sets:
        X_train = get_features(training_data, model_name)
        y_train = training_data[TARGET]

        X_train = X_train[X_test.columns]

        model = make_model(model_name)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = calculate_fairness(
            y_test,
            predictions,
            sex_test,
        )

        result = {
            "Model": model_name,
            "Stage": stage,
            **metrics,
        }

        results.append(result)

        print(f"\n{stage}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
