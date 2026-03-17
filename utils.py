"""
utils.py
========
Configuration, feature engineering, and model helpers for the incremental
temperature forecasting pipeline.

Leakage audit (completed):
  REMOVED  temp_roll3_mean    -- rolling avg of TARGET, corr=0.9986
  REMOVED  temp_roll7_mean    -- rolling avg of TARGET, corr=0.9516
  REMOVED  dewspread          -- (temp - dew_point) encodes target directly
  REMOVED  humidity_temp_ratio-- (humidity / temp) encodes target directly
  REMOVED  apparent_temperature-- derived from temperature
  REMOVED  dew_point_2m       -- partially derived from temperature
  REMOVED  visibility         -- near-zero correlation (0.019)
  REMOVED  snowfall           -- all-zero in Hyderabad dataset
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TARGET = "temperature_2m (°C)"

FEATURE_COLS = [
    # Raw meteorological (from Open-Meteo at prediction time)
    "relative_humidity_2m (%)",
    "precipitation (mm)",
    "rain (mm)",
    "pressure_msl (hPa)",
    "cloud_cover (%)",
    "wind_speed_10m (km/h)",
    "wind_gusts_10m (km/h)",
    "wind_dir_sin",
    "wind_dir_cos",
    # Lag features — legitimate past values, zero leakage
    "temp_lag1",
    "temp_lag2",
    "temp_lag3",
    "humidity_lag1",
    "pressure_lag1",
    # Rolling stats of NON-target features only (safe)
    "humidity_roll3",
    "pressure_roll3",
    # Rate-of-change deltas
    "temp_delta",
    "humidity_delta",
    "pressure_delta",
    "wind_speed_delta",
    # Binary / count derived features
    "is_raining",
    "precipitation_roll6",
    # Temporal encodings
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "is_weekend",
    "is_day",
]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

PYCARET_MODEL_PATH     = MODEL_DIR / "pycaret_best_model"
INCREMENTAL_MODEL_PATH = MODEL_DIR / "incremental_sgd_model.pkl"
SCALER_PATH            = MODEL_DIR / "scaler.pkl"
FEATURE_STATE_PATH     = MODEL_DIR / "feature_state.pkl"


# ---------------------------------------------------------------------------
# DATA UTILITIES
# ---------------------------------------------------------------------------
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load CSV, parse timestamps, sort chronologically, drop junk columns."""
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df.dropna(subset=FEATURE_COLS + [TARGET], inplace=True)
    print(f"[Data] Loaded {len(df):,} rows from {csv_path}")
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.20):
    """
    Chronological train/test split — never shuffle time-series data.

    Args:
        df:        Full sorted DataFrame.
        test_frac: Fraction of rows reserved as test (default 20%).

    Returns:
        (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"[Split] Train: {len(train):,}  |  Test: {len(test):,}")
    return train, test


# ---------------------------------------------------------------------------
# FEATURE STATE MANAGER  (rolling buffer for live inference)
# ---------------------------------------------------------------------------
class FeatureStateManager:
    """
    Maintains a rolling window of the last WINDOW observations so that
    lag and rolling features can be reconstructed from live sensor data.

    Cold-start behaviour:
        On the first few requests the history buffer is short (<4 entries).
        Lag features that would fall outside the buffer are back-filled with
        the earliest available value rather than left as NaN, ensuring the
        model always receives a finite input vector.

    At inference time the ESP32 sends: temperature, humidity, light, heatIndex
    Merged with Open-Meteo context:    pressure, wind, precipitation, etc.

    Usage:
        fsm = FeatureStateManager.load()
        feats = fsm.update(obs_dict)
        fsm.save()
    """
    WINDOW = 6  # max look-back (precipitation_roll6)

    def __init__(self):
        self.history = []

    def update(self, obs: dict) -> dict:
        """Append one observation; return enriched feature dict."""
        self.history.append(obs)
        if len(self.history) > self.WINDOW:
            self.history.pop(0)
        return self._compute_features()

    def save(self):
        joblib.dump(self, FEATURE_STATE_PATH)

    @classmethod
    def load(cls):
        if FEATURE_STATE_PATH.exists():
            return joblib.load(FEATURE_STATE_PATH)
        return cls()

    # -- private helpers -----------------------------------------------------

    def _get(self, key: str, idx: int = -1, default=None):
        """
        Safe look-up into history with index and default.
        If default is None, falls back to the earliest available value for
        that key so lag features are never NaN during cold-start.
        """
        try:
            val = self.history[idx][key]
            return val if val is not None else self._earliest(key, default)
        except (IndexError, KeyError):
            return self._earliest(key, default)

    def _earliest(self, key: str, fallback=np.nan):
        """Return the oldest value in history for key, or fallback."""
        for obs in self.history:
            v = obs.get(key)
            if v is not None:
                return v
        return 0.0 if fallback is None else fallback

    def _compute_features(self) -> dict:
        h = self.history
        n = len(h)

        # Build series — back-fill NaN with earliest available value
        def _series(key):
            vals = [x.get(key, np.nan) for x in h]
            # Forward-fill any gaps using pandas
            s = pd.Series(vals).ffill().bfill().fillna(0.0)
            return s.tolist()

        hum_series  = _series("humidity")
        pres_series = _series("pressure")
        temp_series = _series("temperature")

        now = datetime.now()

        # Helper: get with cold-start back-fill
        def g(key, idx=-1, default=0.0):
            return self._get(key, idx, default)

        # For lag features: if history is shorter than needed, use the
        # earliest available temperature / humidity / pressure reading
        def lag_temp(steps_back):
            """Get temperature from `steps_back` positions ago (1-indexed)."""
            idx = -(steps_back + 1)   # -2 for lag1, -3 for lag2, -4 for lag3
            try:
                return temp_series[idx]
            except IndexError:
                return temp_series[0]  # cold-start: reuse earliest

        def lag_hum(steps_back):
            idx = -(steps_back + 1)
            try:
                return hum_series[idx]
            except IndexError:
                return hum_series[0]

        def lag_pres(steps_back):
            idx = -(steps_back + 1)
            try:
                return pres_series[idx]
            except IndexError:
                return pres_series[0]

        cur_temp  = temp_series[-1]
        prev_temp = temp_series[-2] if n >= 2 else cur_temp
        cur_hum   = hum_series[-1]
        prev_hum  = hum_series[-2] if n >= 2 else cur_hum
        cur_pres  = pres_series[-1]
        prev_pres = pres_series[-2] if n >= 2 else cur_pres
        cur_wind  = g("wind_speed")
        prev_wind = self._get("wind_speed", -2, 0.0) if n >= 2 else cur_wind

        return {
            # Raw meteorological
            "relative_humidity_2m (%)": cur_hum,
            "precipitation (mm)":       g("precipitation", default=0.0),
            "rain (mm)":                g("rain",          default=0.0),
            "pressure_msl (hPa)":       cur_pres,
            "cloud_cover (%)":          g("cloud_cover",   default=0.0),
            "wind_speed_10m (km/h)":    cur_wind,
            "wind_gusts_10m (km/h)":    g("wind_gusts",    default=0.0),
            "wind_dir_sin":             g("wind_dir_sin",  default=0.0),
            "wind_dir_cos":             g("wind_dir_cos",  default=1.0),
            # Lag features — back-filled on cold start, never NaN
            "temp_lag1":     lag_temp(1),
            "temp_lag2":     lag_temp(2),
            "temp_lag3":     lag_temp(3),
            "humidity_lag1": lag_hum(1),
            "pressure_lag1": lag_pres(1),
            # Rolling stats of non-target features
            "humidity_roll3":  float(np.mean(hum_series[-3:])),
            "pressure_roll3":  float(np.mean(pres_series[-3:])),
            # Deltas — zero on cold start (no previous reading to diff)
            "temp_delta":       cur_temp  - prev_temp,
            "humidity_delta":   cur_hum   - prev_hum,
            "pressure_delta":   cur_pres  - prev_pres,
            "wind_speed_delta": cur_wind  - prev_wind,
            # Derived
            "is_raining":          int(g("precipitation", default=0.0) > 0),
            "precipitation_roll6": sum(x.get("precipitation", 0.0) for x in h[-6:]),
            # Temporal
            "hour":       now.hour,
            "dayofweek":  now.weekday(),
            "month":      now.month,
            "dayofyear":  now.timetuple().tm_yday,
            "is_weekend": int(now.weekday() >= 5),
            "is_day":     int(6 <= now.hour < 20),
        }


# ---------------------------------------------------------------------------
# INFERENCE & ONLINE UPDATE HELPERS
# ---------------------------------------------------------------------------
def _safe_feature_array(feature_dict: dict) -> np.ndarray:
    """
    Build a (1, n_features) float array from feature_dict.
    Replaces any remaining NaN/inf with 0.0 as a final safety net —
    this should not occur after FeatureStateManager's cold-start back-fill,
    but guards against unexpected missing Open-Meteo fields at runtime.
    """
    x = np.array([feature_dict.get(c, 0.0) for c in FEATURE_COLS], dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.reshape(1, -1)


def predict_temperature(feature_dict: dict, scaler, model) -> float:
    """Feature dict -> finite numpy array -> scaled -> model prediction (°C)."""
    x = _safe_feature_array(feature_dict)
    return float(model.predict(scaler.transform(x))[0])


def update_model_online(feature_dict: dict, true_temp: float, scaler, model) -> None:
    """
    One step of incremental learning with confirmed ground-truth temperature.
    Updates SGDRegressor via partial_fit() and persists weights to disk.
    """
    x = _safe_feature_array(feature_dict)
    model.partial_fit(scaler.transform(x), [true_temp])
    joblib.dump(model, INCREMENTAL_MODEL_PATH)