import warnings
warnings.filterwarnings("ignore")  # silences XGBoost/sklearn deprecation chatter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('data/combined_statslogExcluded_SMOTE_GenderBalancedWithCTGAN.csv')
print(df.columns)

# ===========================================================================
# SHAP XGBoost [Clinical]
# ===========================================================================
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=115, max_depth=6, learning_rate=0.0547, subsample=0.5715,
    colsample_bytree=0.5422, gamma=1.5894, reg_alpha=0.9968, reg_lambda=0.8258,
    random_state=123,
)
model.fit(X_train, y_train)

# Step 2: Explain model predictions using SHAP
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
shap_values = explainer(X_test)

# Step 3: Plot SHAP summary
shap.summary_plot(shap_values, X_test)  # global impact

print(X_test.iloc[[12]])
print(X_test.iloc[[9]])
print(model.predict(X_test.iloc[[12]]))
print(model.predict(X_test.iloc[[9]]))

# Pick a random index from X_test
random_idx = 12

# Plot the explanation for the selected patient
shap.plots.waterfall(shap_values[random_idx], max_display=11)

# Pick a random index from X_test
random_idx = 9

# Plot the explanation for the selected patient
shap.plots.waterfall(shap_values[random_idx], max_display=11)

# Bar plot of mean(|SHAP value|) per feature
shap.plots.bar(shap_values, max_display=11)


# ===========================================================================
# SHAP RF [Non-Clinical]
# ===========================================================================
reduced_df = df.drop(columns=['chol', 'restecg', 'oldpeak', 'slope'], axis=1)
X = reduced_df.drop('target', axis=1)
y = reduced_df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

# Fit Random Forest
model = RandomForestClassifier(
    n_estimators=110, max_depth=20, min_samples_split=4,
    min_samples_leaf=8, max_features='sqrt', bootstrap=True,
    random_state=123,
)
model.fit(X_train, y_train)


def f(x):
    return model.predict_proba(x)[:, 1]


med = X_train.median().to_frame().T  # 1-row DataFrame, keeps column names

explainer = shap.Explainer(f, med)
shap_values = explainer(X_test.iloc[0:1000, :])

print(model.predict(X_test.iloc[[12]]))
print(model.predict(X_test.iloc[[9]]))

shap.plots.waterfall(shap_values[12])

shap.plots.waterfall(shap_values[9])

# Bar plot of mean(|SHAP value|) per feature
shap.plots.bar(shap_values, max_display=11)


# ===========================================================================
# DiCE XGBoost [Clinical]
# ===========================================================================

import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers

# Re-derive the CLINICAL (full 11-feature) split -- X_train/X_test currently
# point at the reduced non-clinical split from the SHAP RF section above.
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

print(df.dtypes)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=115, max_depth=6, learning_rate=0.0547, subsample=0.5715,
    colsample_bytree=0.5422, gamma=1.5894, reg_alpha=0.9968, reg_lambda=0.8258,
    enable_categorical=True,
    random_state=123,
)
model.fit(X_train, y_train)

print(df.columns)

# Define the target
target = 'target'

# List your continuous and categorical features
continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
categorical_features = []

# DiCE data object
data_dice = dice_ml.Data(
    dataframe=df,
    continuous_features=continuous_features,
    outcome_name=target,
)

# Wrap trained model (sklearn RandomForestClassifier or others)
model_dice = dice_ml.Model(model=model, backend='sklearn')

# DiCE explainer
exp = Dice(data_dice, model_dice)

# Choose a sample from test set
sample = X_test.iloc[[0]]

print("Original prediction:", model.predict(sample))  # should be 0 or 1

# Generate counterfactuals -- age and sex held constant, matching the paper's
# stated methodology ("Age and Sex were held constant to ensure physiological
# plausibility", Table 4 caption). random_seed fixes DiCE's own internal search
# randomness, which is separate from (and not controlled by) the model's own
# random_state -- without it, DiCE explores a different valid counterfactual
# set on every run even when the underlying model is fully deterministic.
features_to_vary = [c for c in X_train.columns if c not in ['age', 'sex']]
dice_exp = exp.generate_counterfactuals(
    sample, total_CFs=6, desired_class="opposite",
    features_to_vary=features_to_vary, random_seed=123,
)

# Display them -- printed directly instead of visualize_as_dataframe(), which requires
# IPython.display and is meant for Jupyter/notebook sessions, not a plain script run.
print(dice_exp.cf_examples_list[0].final_cfs_df)


# ===========================================================================
# DiCE RF [Non-Clinical]
# ===========================================================================
reduced_df = df.drop(columns=['chol', 'restecg', 'slope', 'oldpeak'], axis=1)
X = reduced_df.drop('target', axis=1)
y = reduced_df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=123
)

rfc_clf = RandomForestClassifier(
    n_estimators=110, max_depth=20, min_samples_split=4, min_samples_leaf=8,
    max_features='sqrt', bootstrap=True,
    random_state=123,
)
rfc_clf.fit(X_train, y_train)

print(reduced_df.columns)

# Define the target
target = 'target'

# List your continuous and categorical features
continuous_features = ['age', 'sex', 'cp', 'trestbps', 'fbs', 'thalach', 'exang']
categorical_features = []

# DiCE data object
data_dice = dice_ml.Data(
    dataframe=reduced_df,
    continuous_features=continuous_features,
    outcome_name=target,
)

# Wrap trained model (sklearn RandomForestClassifier or others)
model_dice = dice_ml.Model(model=rfc_clf, backend='sklearn')

# DiCE explainer
exp = Dice(data_dice, model_dice)

# Choose a sample from test set
sample = X_test.iloc[[0]]

print("Original prediction:", rfc_clf.predict(sample))  # should be 0 or 1

# Generate counterfactuals -- age and sex held constant, matching the paper's
# stated methodology ("Age and Sex were held constant to ensure physiological
# plausibility", Table 4 caption). random_seed fixes DiCE's own internal search
# randomness, which is separate from (and not controlled by) the model's own
# random_state -- without it, DiCE explores a different valid counterfactual
# set on every run even when the underlying model is fully deterministic.
features_to_vary = [c for c in X_train.columns if c not in ['age', 'sex']]
dice_exp = exp.generate_counterfactuals(
    sample, total_CFs=6, desired_class="opposite",
    features_to_vary=features_to_vary, random_seed=123,
)

# Display them -- printed directly instead of visualize_as_dataframe(), which requires
# IPython.display and is meant for Jupyter/notebook sessions, not a plain script run.
print(dice_exp.cf_examples_list[0].final_cfs_df)