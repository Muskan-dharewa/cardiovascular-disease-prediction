
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from dowhy import CausalModel


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_PATH = (
    "data/combined_statslogExcluded_SMOTE_GenderBalancedWithCTGAN.csv"
)

TARGET = "target"
ATE_THRESHOLD = 0.1
TEST_SIZE = 0.20
RANDOM_STATE = 123

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope"
]


# ---------------------------------------------------------
# STEP 1: IDENTIFY POTENTIAL CONFOUNDERS
# Paper: Random Forest feature importance; cumulative
# importance should reach at least 80%.
# ---------------------------------------------------------


def select_confounders(data, treatment, random_state=123):

    X = data.drop(columns=[treatment, TARGET], errors="ignore")
    y = data[treatment]

    # Use a classifier for categorical treatments
    # and a regressor for continuous treatments.
    discrete_features = {"sex", "cp", "fbs", "restecg", "exang", "slope"}

    if treatment in discrete_features:
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )

    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    total_importance = importance_df["importance"].sum()

    if total_importance == 0:
        return []

    importance_df["importance"] /= total_importance
    importance_df["cumulative_importance"] = (
        importance_df["importance"].cumsum()
    )

    # Include the feature that reaches or crosses 80%.
    cutoff = np.searchsorted(
        importance_df["cumulative_importance"].to_numpy(),
        0.80,
        side="left"
    )

    return importance_df.loc[
        :min(cutoff, len(importance_df) - 1),
        "feature"
    ].tolist()


# ---------------------------------------------------------
# STEP 2: ESTIMATE ATE USING BACKDOOR ADJUSTMENT
# ---------------------------------------------------------

def estimate_ate(data, treatment):

    confounders = select_confounders(data, treatment)

    print(f"\nTreatment: {treatment}")
    print("Potential confounders:", confounders)

    if not confounders:
        print("No confounders selected; skipping treatment.")
        return np.nan

    causal_model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=TARGET,
        common_causes=confounders
    )

    estimand = causal_model.identify_effect()

    estimate = causal_model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression"
    )

    ate = float(estimate.value)

    print(f"Estimated ATE: {ate:.6f}")

    return ate


# ---------------------------------------------------------
# STEP 3: SELECT FEATURES USING THE ATE THRESHOLD
# ---------------------------------------------------------

def select_causal_features(data):

    ate_results = {}

    print("\n" + "=" * 65)
    print("CAUSAL INFERENCE: AVERAGE TREATMENT EFFECTS")
    print("=" * 65)

    for feature in FEATURES:
        ate_results[feature] = estimate_ate(data, feature)

    ate_df = pd.DataFrame({
        "feature": list(ate_results.keys()),
        "ATE": list(ate_results.values())
    })

    ate_df["absolute_ATE"] = ate_df["ATE"].abs()

    selected_features = ate_df.loc[
        ate_df["absolute_ATE"] >= ATE_THRESHOLD,
        "feature"
    ].tolist()

    print("\nATE SUMMARY")
    print(ate_df.to_string(index=False))

    print(f"\nATE threshold: |ATE| >= {ATE_THRESHOLD}")
    print("Selected causal features:", selected_features)

    if not selected_features:
        raise ValueError(
            "No features met the ATE threshold. "
            "Check the ATE estimates and data."
        )

    return selected_features, ate_df


# ---------------------------------------------------------
# STEP 4: TRAIN AND EVALUATE CGEVC
# ---------------------------------------------------------

def train_cgevc(data, selected_features):

    X = data[selected_features].copy()
    y = data[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Scale/impute where required by distance- and
    # coefficient-based models. Fit preprocessing on train only.
    xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

    lgbm = LGBMClassifier(
        random_state=42,
        verbosity=-1
    )

    rf = RandomForestClassifier(
        random_state=42
    )

    logreg = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    knn = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

    # Soft-voting ensemble described in the paper.
    cgevc = VotingClassifier(
        estimators=[
            ("xgb", xgb),
            ("lgbm", lgbm),
            ("rf", rf),
            ("logreg", logreg),
            ("knn", knn)
        ],
        voting="soft"
    )

    cgevc.fit(X_train, y_train)

    y_pred = cgevc.predict(X_test)
    y_prob = cgevc.predict_proba(X_test)[:, 1]

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, y_pred, average="macro"
        ),
        "Weighted F1": f1_score(
            y_test, y_pred, average="weighted"
        ),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    print("\n" + "=" * 65)
    print("CGEVC PERFORMANCE")
    print("=" * 65)

    print("Selected features:", selected_features)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f} ({value * 100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

    return cgevc, results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[required_columns].dropna().copy()

    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(df[TARGET].value_counts())

    selected_features, ate_df = select_causal_features(df)

    # Use the ATE results already calculated by the script.
    plot_df = ate_df.dropna(subset=["ATE"]).copy()
    plot_df = plot_df.sort_values("ATE")

    colors = [
        "steelblue" if abs(value) >= 0.1 else "lightgray"
        for value in plot_df["ATE"]
    ]

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["feature"], plot_df["ATE"], color=colors)
    plt.axvline(0.1, linestyle="--", color="red",
                label="Selection threshold (+0.1)")
    plt.axvline(-0.1, linestyle="--", color="red",
                label="Selection threshold (-0.1)")
    plt.axvline(0, color="black", linewidth=0.8)

    plt.xlabel("Average Treatment Effect (ATE)")
    plt.ylabel("Feature")
    plt.title("CGEVC: Average Treatment Effect by Feature")
    plt.legend()
    plt.tight_layout()
    plt.show()

    model, results = train_cgevc(df, selected_features)

    print("\nCGEVC experiment completed.")