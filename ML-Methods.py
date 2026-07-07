import os
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    message=".*'multi_class' was deprecated in version 1.5.*",
)

# If you have xgboost installed; otherwise comment this import and the model below.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ----------------------------
# Paths and constants
# ----------------------------
TRAIN_CSV = '/lclhome/zsaeed/MAPS/data/cell_phenotyping/train.csv'
VALID_CSV = '/lclhome/zsaeed/MAPS/data/cell_phenotyping/valid.csv'
LABEL_COL = 'cell_label'
NUM_FEATURES = 50         # sanity check
NUM_CLASSES = 16          # used for AUC labels
RESULTS_DIR = './results_classical'  # separate from FFNN logs
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_standardize(train_csv, valid_csv, label_col):
    """Load CSVs, split X/y, and apply train-based standardization to both."""
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    assert label_col in train_df.columns, f"Label column {label_col} not in train.csv"
    assert label_col in valid_df.columns, f"Label column {label_col} not in valid.csv"

    feature_cols = [c for c in train_df.columns if c != label_col]
    if len(feature_cols) != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, found {len(feature_cols)}")

    X_train = train_df[feature_cols].astype(np.float64).values
    y_train = train_df[label_col].astype(int).values
    X_valid = valid_df[feature_cols].astype(np.float64).values
    y_valid = valid_df[label_col].astype(int).values

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_std = scaler.fit_transform(X_train)
    X_valid_std = scaler.transform(X_valid)

    return X_train_std, y_train, X_valid_std, y_valid, feature_cols


def compute_metrics(y_true, y_pred, y_proba=None, num_classes=NUM_CLASSES):
    """Compute the same metrics as the FFNN trainer."""
    acc = accuracy_score(y_true, y_pred)

    # Multi-class AUC (one-vs-one), if probabilities are available
    if y_proba is not None:
        try:
            auc = roc_auc_score(
                y_true,
                y_proba,
                multi_class='ovo',
                labels=list(range(num_classes))
            )
        except Exception:
            auc = np.nan
    else:
        auc = np.nan

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    return {
        'acc': acc,
        'auc': auc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }


def run_models(X_train, y_train, X_valid, y_valid):
    """Train and evaluate several classical ML models on the same data."""

    results = []
    start = time.time()
    # 1) Logistic Regression (multinomial)
    logreg = LogisticRegression(
        multi_class='multinomial',
        solver='saga',
        max_iter=500,
        n_jobs=-1,
        verbose=0
    )
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_valid)
    y_proba = logreg.predict_proba(X_valid)
    metrics = compute_metrics(y_valid, y_pred, y_proba)
    metrics['model'] = 'LogisticRegression'
    results.append(metrics)
    fit_time = time.time() - start
    print(f"LogisticRegression fit_time: {fit_time:.3f} seconds")

    # 2) Linear SVM (no probabilities by default)
    start = time.time()
    svm = LinearSVC(
        C=1.0,
        max_iter=5000
    )
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_valid)
    # No predict_proba -> pass None for AUC
    metrics = compute_metrics(y_valid, y_pred, y_proba=None)
    metrics['model'] = 'LinearSVM'
    results.append(metrics)
    fit_time = time.time() - start
    print(f"SVM fit_time: {fit_time:.3f} seconds")

    # 3) Random Forest
    start = time.time()
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=0
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_valid)
    y_proba = rf.predict_proba(X_valid)
    metrics = compute_metrics(y_valid, y_pred, y_proba)
    metrics['model'] = 'RandomForest'
    results.append(metrics)
    fit_time = time.time() - start
    print(f"Random Forest fit_time: {fit_time:.3f} seconds")

    # 4) Decision Trees
    start = time.time()
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=0
    )
    et.fit(X_train, y_train)
    y_pred = et.predict(X_valid)
    y_proba = et.predict_proba(X_valid)
    metrics = compute_metrics(y_valid, y_pred, y_proba)
    metrics['model'] = 'ExtraTrees'
    results.append(metrics)
    fit_time = time.time() - start
    print(f"Decision Tree fit_time: {fit_time:.3f} seconds")

    # 5) XGBoost (if available)
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=NUM_CLASSES,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_valid)
        y_proba = xgb.predict_proba(X_valid)
        metrics = compute_metrics(y_valid, y_pred, y_proba)
        metrics['model'] = 'XGBoost'
        results.append(metrics)

    return pd.DataFrame(results)


if __name__ == '__main__':
    print("Loading data and applying same standardization as FFNN...")
    X_train, y_train, X_valid, y_valid, feat_cols = load_and_standardize(
        TRAIN_CSV, VALID_CSV, LABEL_COL
    )

    print("Training and evaluating classical ML baselines...")
    df_results = run_models(X_train, y_train, X_valid, y_valid)

    # Save benchmark results
    out_csv = os.path.join(RESULTS_DIR, 'benchmark_classical_models.csv')
    df_results.to_csv(out_csv, index=False)
    print(f"Saved benchmark results to {out_csv}\n")
    print(df_results.to_string(index=False, float_format=lambda x: f'{x:.4f}'))