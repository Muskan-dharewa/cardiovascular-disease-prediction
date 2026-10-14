import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from scipy.stats import t
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.utils import shuffle
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

NON_CLINICAL_DROP_COLS = ["chol", "restecg", "slope", "oldpeak"]
SHUFFLE_SEED, SMOTE_SEED, SPLIT_SEED, TUNE_SEED = 123, 42, 123, 0


RESULTS = {}      # (phase, feature_set, model) -> (acc, auc, f1)
BEST_PARAMS = {}  # (feature_set, model) -> dict


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def get_ci(values, confidence=0.95):
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    se = np.std(values, ddof=1) / np.sqrt(n)
    h = t.ppf((1 + confidence) / 2.0, n - 1) * se
    return mean, mean - h, mean + h


def make_factory(name, params=None):
    """Return f(seed) -> fresh unfitted model."""
    params = dict(params or {})
    if name == "XGBoost":
        return lambda seed: XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                          random_state=seed, n_jobs=1, **params)
    if name == "LightGBM":
        return lambda seed: LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, **params)
    if name == "Random Forest":
        return lambda seed: RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    raise ValueError(name)


def evaluate(phase, fs_label, name, factory, X_tr, y_tr, X_te, y_te, n_iter, tag=None):
    """Refit n_iter times (seed = iteration index) and report mean + 95% CI."""
    acc, auc, f1 = [], [], []
    for seed in range(n_iter):
        model = factory(seed)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        acc.append(accuracy_score(y_te, pred))
        auc.append(roc_auc_score(y_te, proba))
        f1.append(f1_score(y_te, pred))
    shown = tag or name
    print(f"\n--- {shown} -- {phase} ({fs_label}) ---")
    for lab, vals in (("ACC", acc), ("AUC", auc), ("F1 ", f1)):
        m, lo, hi = get_ci(vals)
        print(f"{lab}: {m:.4f} [{lo:.4f}, {hi:.4f}]")
    RESULTS[(phase, fs_label, tag or name)] = (np.mean(acc), np.mean(auc), np.mean(f1))


def load_and_impute(path):
    df = pd.read_csv(path)
    feats = [c for c in df.columns if c != "target"]
    n_nan = int(df[feats].isna().sum().sum())
    if n_nan:
        df[feats] = SimpleImputer(strategy="median").fit_transform(df[feats])
    return df, n_nan


def _cv_f1(model, X, y, cv):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=TUNE_SEED)
    return cross_val_score(model, X, y, cv=skf, scoring="f1").mean()


def tune_xgb_bayesian(X, y, n_trials, cv, label):
    pbounds = {
        "n_estimators": (50, 200), "max_depth": (3, 10), "learning_rate": (0.01, 0.3),
        "subsample": (0.5, 1.0), "colsample_bytree": (0.5, 1.0), "gamma": (0.0, 5.0),
        "reg_alpha": (0.0, 1.0), "reg_lambda": (0.0, 1.0),
    }

    def to_params(p):
        q = dict(p)
        q["n_estimators"] = int(round(q["n_estimators"]))
        q["max_depth"] = int(round(q["max_depth"]))
        return q

    def objective(**p):
        model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                              random_state=TUNE_SEED, n_jobs=1, **to_params(p))
        return _cv_f1(model, X, y, cv)

    init = min(10, max(2, n_trials // 3))
    opt = BayesianOptimization(f=objective, pbounds=pbounds, random_state=TUNE_SEED, verbose=0)
    opt.maximize(init_points=init, n_iter=max(0, n_trials - init))
    best = to_params(opt.max["params"])
    print(f"[{label}] XGBoost  (Bayesian Opt.)  best CV-F1={opt.max['target']:.4f}\n    {json.dumps(best)}")
    return best


def tune_lgbm_optuna(X, y, n_trials, cv, label):
    def objective(tr):
        model = LGBMClassifier(
            n_estimators=tr.suggest_int("n_estimators", 50, 200),
            max_depth=tr.suggest_int("max_depth", 3, 10),
            learning_rate=tr.suggest_float("learning_rate", 0.01, 0.3),
            subsample=tr.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0),
            num_leaves=tr.suggest_int("num_leaves", 15, 150),
            min_child_samples=tr.suggest_int("min_child_samples", 5, 50),
            reg_alpha=tr.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda=tr.suggest_float("reg_lambda", 0.0, 1.0),
            random_state=TUNE_SEED, n_jobs=1, verbose=-1)
        return _cv_f1(model, X, y, cv)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
    study.optimize(objective, n_trials=n_trials)
    print(f"[{label}] LightGBM (Optuna)         best CV-F1={study.best_value:.4f}\n    {json.dumps(study.best_params)}")
    return dict(study.best_params)


def tune_rf_optuna(X, y, n_trials, cv, label):
    def objective(tr):
        model = RandomForestClassifier(
            n_estimators=tr.suggest_int("n_estimators", 50, 300),
            max_depth=tr.suggest_int("max_depth", 3, 20),
            min_samples_split=tr.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=tr.suggest_int("min_samples_leaf", 1, 10),
            max_features=tr.suggest_categorical("max_features", ["sqrt", "log2", None]),
            bootstrap=tr.suggest_categorical("bootstrap", [True, False]),
            random_state=TUNE_SEED, n_jobs=1)
        return _cv_f1(model, X, y, cv)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
    study.optimize(objective, n_trials=n_trials)
    print(f"[{label}] RandomForest (Optuna)     best CV-F1={study.best_value:.4f}\n    {json.dumps(study.best_params)}")
    return dict(study.best_params)


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def experiment_a(df_clev, df_comb, n_iter):
    for fs_label, drop in (("Clinical", []), ("Non-Clinical", NON_CLINICAL_DROP_COLS)):
        print("\n" + "=" * 70)
        print(f"EXPERIMENT A: GENERALIZATION TEST -- {fs_label.upper()} FEATURES")
        print("(train: Cleveland only | test: Multi-Region | default hyper-parameters)")
        print("=" * 70)
        tr, te = df_clev.drop(columns=drop), df_comb.drop(columns=drop)
        X_tr, y_tr = tr.drop(columns="target"), tr["target"]
        X_te, y_te = te.drop(columns="target"), te["target"]
        for name in ("XGBoost", "LightGBM", "Random Forest"):
            evaluate("Generalization", fs_label, name, make_factory(name),
                     X_tr, y_tr, X_te, y_te, n_iter)

def plot_smote_class_balance(original_df, smote_df):
    import matplotlib.pyplot as plt

    original_counts = original_df["target"].value_counts().sort_index()
    smote_counts = smote_df["target"].value_counts().sort_index()

    labels = ["Non-CVD (0)", "CVD (1)"]
    original_values = [original_counts.get(0, 0), original_counts.get(1, 0)]
    smote_values = [smote_counts.get(0, 0), smote_counts.get(1, 0)]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, original_values, width, label="Original (918)")
    bars2 = ax.bar(x + width / 2, smote_values, width, label="After SMOTE (1016)")

    ax.set_title("Class balance before and after SMOTE")
    ax.set_ylabel("Number of patients")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(bars1, padding=3)
    ax.bar_label(bars2, padding=3)

    plt.tight_layout()
    plt.show()


def build_smote_dataset(path):
    df = shuffle(pd.read_csv(path), random_state=SHUFFLE_SEED).reset_index(drop=True)
    X, y = df.drop(columns="target"), df["target"]
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
    Xr, yr = SMOTE(random_state=SMOTE_SEED).fit_resample(X, y)   # applied ONCE
    out = pd.DataFrame(Xr, columns=X.columns)
    out["target"] = yr.values
    print("\nSMOTE-balanced shape:", out.shape, "| class counts:",
          out["target"].value_counts().to_dict())
    out.to_csv("data/combined_statslogExcluded_SMOTE.csv", index=False)
    plot_smote_class_balance(df, out)
    print("Saved: data/combined_statslogExcluded_SMOTE.csv")
    return out


def experiment_b(raw_path, n_iter, n_trials, cv, rf_tuning):
    df_smote = build_smote_dataset(raw_path)
    for fs_label, drop in (("Clinical", []), ("Non-Clinical", NON_CLINICAL_DROP_COLS)):
        print("\n" + "=" * 70)
        print(f"EXPERIMENT B: FINAL MODELING -- {fs_label.upper()} FEATURES (SMOTE-balanced)")
        print(f"(80/20 stratified split; tuning on train only: {n_trials} evaluations, {cv}-fold CV, F1)")
        print("=" * 70)
        d = df_smote.drop(columns=drop)
        Xf, yf = d.drop(columns="target"), d["target"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xf, yf, test_size=0.2, stratify=yf, random_state=SPLIT_SEED)

        xgb_p = tune_xgb_bayesian(X_tr, y_tr, n_trials, cv, fs_label)
        evaluate("Final", fs_label, "XGBoost", make_factory("XGBoost", xgb_p),
                 X_tr, y_tr, X_te, y_te, n_iter)

        lgbm_p = tune_lgbm_optuna(X_tr, y_tr, n_trials, cv, fs_label)
        evaluate("Final", fs_label, "LightGBM", make_factory("LightGBM", lgbm_p),
                 X_tr, y_tr, X_te, y_te, n_iter)

        BEST_PARAMS[(fs_label, "XGBoost")] = xgb_p
        BEST_PARAMS[(fs_label, "LightGBM")] = lgbm_p

        if rf_tuning in ("optuna", "both"):
            rf_p = tune_rf_optuna(X_tr, y_tr, n_trials, cv, fs_label)
            BEST_PARAMS[(fs_label, "Random Forest")] = rf_p
            evaluate("Final", fs_label, "Random Forest", make_factory("Random Forest", rf_p),
                     X_tr, y_tr, X_te, y_te, n_iter, tag="Random Forest")
        if rf_tuning == "none":
            evaluate("Final", fs_label, "Random Forest", make_factory("Random Forest"),
                     X_tr, y_tr, X_te, y_te, n_iter, tag="Random Forest")
        elif rf_tuning == "both":
            evaluate("Final", fs_label, "Random Forest", make_factory("Random Forest"),
                     X_tr, y_tr, X_te, y_te, n_iter, tag="Random Forest (no tuning)")
# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=100, help="refits for the 95%% CI")
    ap.add_argument("--n-trials", type=int, default=30, help="tuning evaluations per model")
    ap.add_argument("--cv", type=int, default=10, help="CV folds used while tuning")
    ap.add_argument("--rf-tuning", choices=["optuna", "none", "both"], default="optuna")
    args = ap.parse_args()

    df_clev, _ = load_and_impute("data/cleveland.csv")
    df_comb, n_nan = load_and_impute("data/combined_statslogExcluded.csv")
    print(f"Cleveland: {df_clev.shape} | Multi-Region: {df_comb.shape} "
          f"({n_nan} missing values median-imputed)")

    experiment_a(df_clev, df_comb, args.n_iter)
    experiment_b("data/combined_statslogExcluded.csv", args.n_iter, args.n_trials,
                 args.cv, args.rf_tuning)