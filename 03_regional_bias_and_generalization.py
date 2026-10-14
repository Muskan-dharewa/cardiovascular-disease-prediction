
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

RANDOM_STATE = 123
NON_CLINICAL_DROP_COLS = ["chol", "restecg", "slope", "oldpeak"]

# 1. Load original dataset
df = pd.read_csv("data/combined_statslogExcluded.csv")
print("Dataset shape:", df.shape)

# 2. Create three regional clusters using patient features only
feature_cols = [c for c in df.columns if c != "target"]

X_cluster = SimpleImputer(strategy="median").fit_transform(
    df[feature_cols]
)
X_cluster = StandardScaler().fit_transform(X_cluster)

df["cluster_region"] = AgglomerativeClustering(
    n_clusters=3
).fit_predict(X_cluster)

print("\n--- Regional Cluster Labels ---")
print("Silhouette score:",
      round(silhouette_score(X_cluster, df["cluster_region"]), 4))
print("\nCluster sizes:")
print(df["cluster_region"].value_counts().sort_index())

# 3. Prepare a stratified train/test split
# Cluster labels are kept only for evaluation, not used as predictors.
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    stratify=df["target"],
    random_state=RANDOM_STATE
)

y_train = df.loc[train_idx, "target"]
y_test = df.loc[test_idx, "target"]
test_groups = df.loc[test_idx, "cluster_region"]


def evaluate_model(name, model, columns):
    X = df[columns]

    # Imputation/scaling are fitted only on training data.
    model.fit(X.loc[train_idx], y_train)
    predictions = model.predict(X.loc[test_idx])

    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)

    print("\nOverall test performance:")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, predictions, zero_division=0):.4f}")

    results = pd.DataFrame({
        "target": y_test,
        "prediction": predictions,
        "cluster_region": test_groups
    })

    print("\nPerformance by regional cluster:")
    for cluster in sorted(results["cluster_region"].unique()):
        group = results[results["cluster_region"] == cluster]
        print(f"\nCluster {cluster} (n={len(group)}):")
        print(f"Accuracy : {accuracy_score(group['target'], group['prediction']):.4f}")
        print(f"Precision: {precision_score(group['target'], group['prediction'], zero_division=0):.4f}")
        print(f"Recall   : {recall_score(group['target'], group['prediction'], zero_division=0):.4f}")
        print(f"F1-score : {f1_score(group['target'], group['prediction'], zero_division=0):.4f}")


# 4. Clinical XGBoost model
clinical_features = feature_cols

clinical_model = make_pipeline(
    SimpleImputer(strategy="median"),
    xgb.XGBClassifier(
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
        random_state=RANDOM_STATE
    )
)

evaluate_model(
    "Clinical Model — XGBoost",
    clinical_model,
    clinical_features
)

# 5. Non-clinical Random Forest model
non_clinical_features = [
    c for c in feature_cols if c not in NON_CLINICAL_DROP_COLS
]

non_clinical_model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=110,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=8,
        max_features="sqrt",
        bootstrap=True,
        random_state=RANDOM_STATE
    )
)

evaluate_model(
    "Non-Clinical Model — Random Forest",
    non_clinical_model,
    non_clinical_features
)

print("\nRegional performance evaluation completed.")

# Save regional cluster labels
df.to_csv("data/combined_with_theoretical_region.csv", index=False)
print("\nRegional cluster labels saved to data/combined_with_theoretical_region.csv")
