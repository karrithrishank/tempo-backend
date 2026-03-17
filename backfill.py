"""
backfill.py
===========
Backfills the last N days of historical data from data.csv into all four
Supabase tables:
    sensor_readings    — one row per hour (source='weatherapi', historical)
    hourly_data        — hourly aggregates
    daily_data         — daily aggregates
    model_performance  — prediction vs actual for every hour

Usage:
    python backfill.py                      # last 7 days
    python backfill.py --days 30            # last 30 days
    python backfill.py --days 7 --dry-run   # preview without writing to DB

Prerequisites (run once in Supabase SQL Editor):
    See add_constraints.sql

The script is idempotent — existing rows are skipped, not duplicated.
"""

import argparse
import math
import sys
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backfill")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(val, default=None):
    try:
        v = float(val)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 4)
    except Exception:
        return default


def _build_wind_components(wind_degree: float):
    rad = math.radians(float(wind_degree))
    return math.sin(rad), math.cos(rad)


def _heat_index(T, RH):
    """Simplified Steadman heat index in °C."""
    if T is None or T < 27:
        return T
    hi = (-8.78469475556
          + 1.61139411   * T
          + 2.33854883889 * RH
          - 0.14611605   * T * RH
          - 0.012308094  * T ** 2
          - 0.0164248277778 * RH ** 2
          + 0.002211732  * T ** 2 * RH
          + 0.00072546   * T * RH ** 2
          - 0.000003582  * T ** 2 * RH ** 2)
    return round(hi, 2)


def _chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# ---------------------------------------------------------------------------
# STEP 1 — Load & filter CSV
# ---------------------------------------------------------------------------
def load_csv(csv_path: str, days: int) -> pd.DataFrame:
    log.info(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    cutoff = df["time"].max() - pd.Timedelta(days=days)
    df = df[df["time"] > cutoff].copy().reset_index(drop=True)
    log.info(f"Rows in last {days} days: {len(df):,}  "
             f"({df['time'].min()} → {df['time'].max()})")
    return df


# ---------------------------------------------------------------------------
# STEP 2 — Generate model predictions row-by-row
# ---------------------------------------------------------------------------
def generate_predictions(df: pd.DataFrame, model_dir: str = "models") -> pd.DataFrame:
    from utils import FEATURE_COLS, FeatureStateManager, predict_temperature

    log.info("Loading model + scaler ...")
    model  = joblib.load(f"{model_dir}/incremental_sgd_model.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    fsm    = FeatureStateManager()

    predictions       = []
    feature_snapshots = []

    for _, row in df.iterrows():
        wind_sin, wind_cos = _build_wind_components(row["wind_direction_10m (°)"])
        obs = {
            "temperature":   _safe_float(row["temperature_2m (°C)"]),
            "humidity":      _safe_float(row["relative_humidity_2m (%)"]),
            "pressure":      _safe_float(row["pressure_msl (hPa)"]),
            "wind_speed":    _safe_float(row["wind_speed_10m (km/h)"]),
            "wind_gusts":    _safe_float(row["wind_gusts_10m (km/h)"]),
            "wind_dir_sin":  wind_sin,
            "wind_dir_cos":  wind_cos,
            "precipitation": _safe_float(row["precipitation (mm)"], 0.0),
            "rain":          _safe_float(row["rain (mm)"], 0.0),
            "cloud_cover":   _safe_float(row["cloud_cover (%)"], 0.0),
            "is_day":        int(row["is_day ()"]),
        }
        feats = fsm.update(obs)
        pred  = predict_temperature(feats, scaler, model)
        predictions.append(round(pred, 4))
        feature_snapshots.append(feats)

    df = df.copy()
    df["predicted_temp"]    = predictions
    df["feature_snapshots"] = feature_snapshots

    mae  = round(float(np.mean(np.abs(df["predicted_temp"] - df["temperature_2m (°C)"]))), 4)
    rmse = round(float(np.sqrt(np.mean((df["predicted_temp"] - df["temperature_2m (°C)"]) ** 2))), 4)
    log.info(f"Predictions generated — MAE={mae}°C  RMSE={rmse}°C")
    return df


# ---------------------------------------------------------------------------
# Safe batch insert — skips rows that already exist (by timestamp)
# ---------------------------------------------------------------------------
def _safe_batch_insert(client, table: str, rows: list, ts_field: str, chunk_size: int = 200) -> dict:
    """
    Insert rows in chunks.  Before each chunk, fetch which timestamps already
    exist and skip them — avoids the need for a UNIQUE constraint at the DB
    level while still being idempotent on re-runs.

    Returns:
        id_map  {ts_string -> row_id}  for sensor_readings / model_performance
    """
    id_map       = {}
    total_new    = 0
    total_skip   = 0

    for chunk in _chunk(rows, chunk_size):
        ts_values = [r[ts_field] for r in chunk]

        # Fetch already-existing timestamps in this chunk
        existing_result = (
            client.table(table)
            .select(f"id,{ts_field}")
            .in_(ts_field, ts_values)
            .execute()
        )
        existing_ts = {r[ts_field]: r["id"] for r in existing_result.data}

        # Collect already-existing id mappings
        id_map.update(existing_ts)
        total_skip += len(existing_ts)

        # Filter to only new rows
        new_rows = [r for r in chunk if r[ts_field] not in existing_ts]
        if not new_rows:
            continue

        result = client.table(table).insert(new_rows).execute()
        for rec in result.data:
            id_map[rec[ts_field]] = rec["id"]
        total_new += len(result.data)

    log.info(f"  {table}: {total_new} inserted, {total_skip} already existed — skipped")
    return id_map


def _safe_batch_upsert(client, table: str, rows: list, conflict_col: str, chunk_size: int = 200) -> int:
    """
    Upsert rows using the DB unique constraint.
    Falls back to insert-skip pattern if constraint is missing (catches APIError).
    """
    total = 0
    for chunk in _chunk(rows, chunk_size):
        try:
            result = (
                client.table(table)
                .upsert(chunk, on_conflict=conflict_col)
                .execute()
            )
            total += len(result.data)
        except Exception as exc:
            if "42P10" in str(exc) or "no unique" in str(exc).lower():
                # Constraint missing — fall back to insert-skip
                log.warning(
                    f"  No unique constraint on {table}.{conflict_col} — "
                    "falling back to insert-skip. "
                    "Run add_constraints.sql in Supabase for faster upserts."
                )
                _id_map = _safe_batch_insert(client, table, chunk, conflict_col)
                total += len(_id_map)
            else:
                raise
    log.info(f"  {table}: {total} rows upserted")
    return total


# ---------------------------------------------------------------------------
# STEP 3 — Insert sensor_readings
# ---------------------------------------------------------------------------
def backfill_sensor_readings(df: pd.DataFrame, client, dry_run: bool) -> dict:
    log.info("Backfilling sensor_readings ...")

    rows = []
    for _, row in df.iterrows():
        ts = row["time"]
        T  = _safe_float(row["temperature_2m (°C)"])
        RH = _safe_float(row["relative_humidity_2m (%)"])
        ws, wc = _build_wind_components(row["wind_direction_10m (°)"])

        rows.append({
            "recorded_at":    ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "temperature":    T,
            "humidity":       RH,
            "heat_index":     _heat_index(T or 25.0, RH or 60.0),
            "is_day":         int(row["is_day ()"]),
            "pressure_mb":    _safe_float(row["pressure_msl (hPa)"]),
            "wind_kph":       _safe_float(row["wind_speed_10m (km/h)"]),
            "wind_gust_kph":  _safe_float(row["wind_gusts_10m (km/h)"]),
            "wind_degree":    _safe_float(row["wind_direction_10m (°)"]),
            "wind_dir_sin":   round(ws, 6),
            "wind_dir_cos":   round(wc, 6),
            "precip_mm":      _safe_float(row["precipitation (mm)"], 0.0),
            "rain_mm":        _safe_float(row["rain (mm)"], 0.0),
            "cloud_cover":    _safe_float(row["cloud_cover (%)"], 0.0),
            "condition_text": "",
            "source":         "weatherapi",
            "api_response":   {},
        })

    if dry_run:
        log.info(f"[DRY RUN] Would insert {len(rows)} sensor_readings rows")
        return {r["recorded_at"]: i + 1 for i, r in enumerate(rows)}

    return _safe_batch_insert(client, "sensor_readings", rows, "recorded_at")


# ---------------------------------------------------------------------------
# STEP 4 — Insert model_performance
# ---------------------------------------------------------------------------
def backfill_model_performance(df: pd.DataFrame, id_map: dict, client, dry_run: bool):
    log.info("Backfilling model_performance ...")

    rows = []
    for _, row in df.iterrows():
        ts_str = row["time"].strftime("%Y-%m-%dT%H:%M:%S+00:00")
        rows.append({
            "predicted_at":      ts_str,
            "predicted_temp":    row["predicted_temp"],
            "actual_temp":       _safe_float(row["temperature_2m (°C)"]),
            "sensor_reading_id": id_map.get(ts_str),
            "feature_snapshot":  row["feature_snapshots"],
            "model_version":     "sgd_v1_backfill",
        })

    if dry_run:
        log.info(f"[DRY RUN] Would insert {len(rows)} model_performance rows")
        return

    _safe_batch_insert(client, "model_performance", rows, "predicted_at")


# ---------------------------------------------------------------------------
# STEP 5 — Upsert hourly_data
# ---------------------------------------------------------------------------
def backfill_hourly_data(df: pd.DataFrame, client, dry_run: bool):
    log.info("Backfilling hourly_data ...")

    rows = []
    for _, row in df.iterrows():
        ts = row["time"]
        h_start = ts.replace(minute=0, second=0, microsecond=0)
        h_end   = h_start + timedelta(hours=1)
        ws, wc  = _build_wind_components(row["wind_direction_10m (°)"])
        T       = _safe_float(row["temperature_2m (°C)"])
        RH      = _safe_float(row["relative_humidity_2m (%)"])

        rows.append({
            "hour_start":           h_start.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "hour_end":             h_end.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "avg_temperature":      T,
            "avg_humidity":         RH,
            "avg_heat_index":       _heat_index(T or 25.0, RH or 60.0),
            "avg_pressure_mb":      _safe_float(row["pressure_msl (hPa)"]),
            "avg_wind_kph":         _safe_float(row["wind_speed_10m (km/h)"]),
            "avg_wind_gust_kph":    _safe_float(row["wind_gusts_10m (km/h)"]),
            "total_precip_mm":      _safe_float(row["precipitation (mm)"], 0.0),
            "avg_cloud_cover":      _safe_float(row["cloud_cover (%)"], 0.0),
            "avg_wind_dir_sin":     round(ws, 6),
            "avg_wind_dir_cos":     round(wc, 6),
            "hour_of_day":          ts.hour,
            "day_of_week":          ts.weekday(),
            "month":                ts.month,
            "day_of_year":          ts.timetuple().tm_yday,
            "is_weekend":           ts.weekday() >= 5,
            "is_day":               int(row["is_day ()"]),
            "reading_count":        1,
            "source":               "weatherapi_fallback",
        })

    if dry_run:
        log.info(f"[DRY RUN] Would upsert {len(rows)} hourly_data rows")
        return

    _safe_batch_upsert(client, "hourly_data", rows, "hour_start")


# ---------------------------------------------------------------------------
# STEP 6 — Upsert daily_data
# ---------------------------------------------------------------------------
def backfill_daily_data(df: pd.DataFrame, client, dry_run: bool):
    log.info("Backfilling daily_data ...")

    df = df.copy()
    df["date"] = df["time"].dt.date

    rows = []
    for date, group in df.groupby("date"):
        T  = group["temperature_2m (°C)"].dropna()
        RH = group["relative_humidity_2m (%)"].dropna()
        P  = group["pressure_msl (hPa)"].dropna()
        W  = group["wind_speed_10m (km/h)"].dropna()
        Pr = group["precipitation (mm)"].dropna()
        CC = group["cloud_cover (%)"].dropna()
        dt = datetime(date.year, date.month, date.day)

        rows.append({
            "date":             date.isoformat(),
            "min_temperature":  round(float(T.min()),  4) if len(T) else None,
            "max_temperature":  round(float(T.max()),  4) if len(T) else None,
            "avg_temperature":  round(float(T.mean()), 4) if len(T) else None,
            "avg_humidity":     round(float(RH.mean()),4) if len(RH) else None,
            "avg_pressure_mb":  round(float(P.mean()), 4) if len(P) else None,
            "avg_wind_kph":     round(float(W.mean()), 4) if len(W) else None,
            "total_precip_mm":  round(float(Pr.sum()), 4) if len(Pr) else 0.0,
            "avg_cloud_cover":  round(float(CC.mean()),4) if len(CC) else None,
            "day_of_week":      dt.weekday(),
            "month":            dt.month,
            "day_of_year":      dt.timetuple().tm_yday,
            "is_weekend":       dt.weekday() >= 5,
            "hours_with_data":  len(group),
            "hours_sensor":     0,
            "hours_fallback":   len(group),
        })

    if dry_run:
        log.info(f"[DRY RUN] Would upsert {len(rows)} daily_data rows")
        return

    _safe_batch_upsert(client, "daily_data", rows, "date")


# ---------------------------------------------------------------------------
# SUMMARY REPORT
# ---------------------------------------------------------------------------
def print_summary(df: pd.DataFrame):
    log.info("=" * 60)
    log.info("BACKFILL SUMMARY")
    log.info("=" * 60)
    log.info(f"  Date range    : {df['time'].min()} → {df['time'].max()}")
    log.info(f"  Total hours   : {len(df)}")
    log.info(f"  Total days    : {df['time'].dt.date.nunique()}")
    T = df["temperature_2m (°C)"]
    log.info(f"  Temp range    : {T.min():.1f}°C — {T.max():.1f}°C  avg={T.mean():.1f}°C")
    if "predicted_temp" in df.columns:
        errs = (df["predicted_temp"] - T).abs()
        log.info(f"  Model MAE     : {errs.mean():.4f}°C")
        log.info(f"  Model RMSE    : {np.sqrt((errs**2).mean()):.4f}°C")
        log.info(f"  Max error     : {errs.max():.4f}°C")
    log.info(f"  Rainy hours   : {int((df['precipitation (mm)'] > 0).sum())}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Backfill Supabase tables from historical CSV data."
    )
    parser.add_argument("--csv",     default="data.csv",
                        help="Path to data.csv (default: data.csv)")
    parser.add_argument("--days",    type=int, default=7,
                        help="Days to backfill from end of CSV (default: 7)")
    parser.add_argument("--models",  default="models",
                        help="Model directory (default: models/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing to Supabase")
    args = parser.parse_args()

    log.info(f"Backfill started — days={args.days}  dry_run={args.dry_run}")

    df = load_csv(args.csv, args.days)
    df = generate_predictions(df, args.models)
    print_summary(df)

    if args.dry_run:
        log.info("[DRY RUN] Skipping all Supabase writes.")
        backfill_sensor_readings(df, None, dry_run=True)
        backfill_model_performance(df, {}, None, dry_run=True)
        backfill_hourly_data(df, None, dry_run=True)
        backfill_daily_data(df, None, dry_run=True)
        log.info("[DRY RUN] Complete.")
        return

    try:
        from db_client import get_client
        client = get_client()
        log.info("Supabase client connected.")
    except Exception as exc:
        log.error(f"Could not connect to Supabase: {exc}")
        sys.exit(1)

    id_map = backfill_sensor_readings(df, client, dry_run=False)
    backfill_model_performance(df, id_map, client, dry_run=False)
    backfill_hourly_data(df, client, dry_run=False)
    backfill_daily_data(df, client, dry_run=False)

    log.info("All tables backfilled successfully.")
    print_summary(df)


if __name__ == "__main__":
    main()