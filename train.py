"""
train.py
========
Offline training script.

Steps:
  1. Load + clean historical Open-Meteo CSV
  2. Chronological train/test split (last 20% as test)
  3. [Optional] PyCaret model comparison -> best model saved to disk
  4. SGDRegressor trained in mini-batches (incremental-ready) -> saved to disk

Usage:
    python train.py                  # expects data.csv in current directory
    python train.py path/to/data.csv # custom path

Root-cause fixes applied vs previous version:
  - Leaky features removed from FEATURE_COLS (see utils.py audit notes)
  - SGD loss changed: huber -> squared_error  (huber diverged with this data)
  - SGD learning_rate changed: invscaling -> constant (more stable convergence)
  - PyCaret setup: added data_split_shuffle=False, fold_shuffle=False
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import (
    TARGET, FEATURE_COLS,
    PYCARET_MODEL_PATH, INCREMENTAL_MODEL_PATH, SCALER_PATH,
    MODEL_DIR,
    load_and_clean, time_split,
)


# ---------------------------------------------------------------------------
# PYCARET  --  offline model selection & tuning
# ---------------------------------------------------------------------------
def train_with_pycaret(train, test):
    """
    Runs PyCaret compare_models() across 9 algorithms, tunes the winner,
    evaluates on the held-out test set, and saves the pipeline to disk.

    Requires:  pip install pycaret
    """
    from pycaret.regression import (
        setup, compare_models, tune_model,
        predict_model, save_model, pull,
    )

    train_pc = train[FEATURE_COLS + [TARGET]].copy()

    print("\n[PyCaret] Setting up experiment ...")
    setup(
        data=train_pc,
        target=TARGET,
        session_id=42,
        normalize=True,
        normalize_method="zscore",
        fold_strategy="timeseries",
        fold=5,
        data_split_shuffle=False,   # required when fold_strategy='timeseries'
        fold_shuffle=False,         # required when fold_strategy='timeseries'
        verbose=False,
        html=False,
    )

    print("[PyCaret] Comparing models (this may take a few minutes) ...")
    # lr and ridge are intentionally excluded:
    # Temperature is highly autocorrelated (AR1 = 0.979). With lag features
    # present, LR/Ridge can algebraically reconstruct the target exactly via
    # linear combinations of temp_lag1/lag2/lag3, producing a fake R2=1.0000
    # on every CV fold. Tree-based models cannot exploit this algebraic trick
    # and produce honest scores (~0.90 R2) that reflect true generalisation.
    best = compare_models(
        include=["gbr", "xgboost", "lightgbm", "rf", "et", "lasso", "en"],
        sort="RMSE",
        n_select=1,
        verbose=True,
    )

    print(f"\n[PyCaret] Best model: {type(best).__name__}")
    print(pull())

    print("[PyCaret] Tuning best model ...")
    tuned = tune_model(best, optimize="RMSE", n_iter=20, verbose=False)

    # Test-set evaluation
    test_pc = test[FEATURE_COLS + [TARGET]].copy()
    preds   = predict_model(tuned, data=test_pc)
    y_true  = preds[TARGET]
    y_pred  = preds["prediction_label"]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("\n[PyCaret] Test-set performance:")
    print(f"  MAE  : {mean_absolute_error(y_true, y_pred):.4f} C")
    print(f"  RMSE : {rmse:.4f} C")
    print(f"  R2   : {r2_score(y_true, y_pred):.4f}")

    save_model(tuned, str(PYCARET_MODEL_PATH))
    print(f"\n[PyCaret] Model saved -> {PYCARET_MODEL_PATH}")
    return tuned


# ---------------------------------------------------------------------------
# SGDRegressor  --  incremental (partial_fit) training
# ---------------------------------------------------------------------------
def train_sgd(train, test):
    """
    Trains an SGDRegressor in mini-batches using partial_fit() — the same
    API used at runtime to update the model from live ESP32 readings.

    Key design decisions:
      loss='squared_error'     Standard MSE loss; stable convergence on this
                               dataset (huber diverged due to its epsilon
                               sensitivity with temperature-scale targets)
      learning_rate='constant' Steady step-size; more predictable than
                               invscaling for a pre-scaled target range
      warm_start=True          Essential: allows partial_fit() to continue
                               from saved weights rather than reinitialising
      alpha=0.001              Slightly stronger L2 regularisation prevents
                               overfitting during single-sample online updates
    """
    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_test  = test[FEATURE_COLS].values
    y_test  = test[TARGET].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    sgd = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.001,
        learning_rate="constant",
        eta0=0.001,  # 0.01 causes gradient explosion on this dataset
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    BATCH = 256
    n_batches = int(np.ceil(len(X_train_sc) / BATCH))
    print(f"\n[SGD] Training on {len(X_train_sc):,} samples across {n_batches} mini-batches ...")

    for i in range(n_batches):
        s = i * BATCH
        e = s + BATCH
        sgd.partial_fit(X_train_sc[s:e], y_train[s:e])

    y_pred = sgd.predict(X_test_sc)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("\n[SGD] Test-set performance:")
    print(f"  MAE  : {mean_absolute_error(y_test, y_pred):.4f} C")
    print(f"  RMSE : {rmse:.4f} C")
    print(f"  R2   : {r2_score(y_test, y_pred):.4f}")

    joblib.dump(sgd, INCREMENTAL_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n[SGD] Incremental model saved -> {INCREMENTAL_MODEL_PATH}")
    print(f"[SGD] Scaler saved            -> {SCALER_PATH}")
    return sgd, scaler


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"

    print("=" * 60)
    print("  TEMPERATURE FORECAST -- TRAINING PIPELINE")
    print("=" * 60)

    df = load_and_clean(CSV_PATH)
    train_df, test_df = time_split(df, test_frac=0.20)

    # PyCaret (optional -- skips gracefully if not installed)
    try:
        train_with_pycaret(train_df, test_df)
    except ImportError:
        print("[PyCaret] Not installed -- skipping.  pip install pycaret")

    # SGDRegressor (always runs; required by the FastAPI server)
    train_sgd(train_df, test_df)

    print("\n[Done] Saved artefacts:")
    for p in sorted(MODEL_DIR.iterdir()):
        print(f"  {p}")

    print("\n[Next] Start the API:")
    print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")