"""
db_client.py
============
Supabase database client — all insert / query operations for the
temperature forecasting system.

Uses the supabase-py client (REST + PostgREST under the hood).
Requires environment variables:
    SUPABASE_URL   — e.g. https://xxxx.supabase.co
    SUPABASE_KEY   — service-role key (not anon key, for server-side writes)

Install:
    pip install supabase
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError(
        "SUPABASE_URL and SUPABASE_KEY must be set in .env or environment."
    )

_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_client() -> Client:
    return _client


# ---------------------------------------------------------------------------
# TABLE: sensor_readings
# ---------------------------------------------------------------------------
def insert_sensor_reading(
    *,
    temperature:    Optional[float],
    humidity:       Optional[float],
    heat_index:     Optional[float],
    is_day:         int,
    pressure_mb:    float,
    wind_kph:       float,
    wind_gust_kph:  float,
    wind_degree:    float,
    wind_dir_sin:   float,
    wind_dir_cos:   float,
    precip_mm:      float,
    rain_mm:        float,
    cloud_cover:    float,
    condition_text: str,
    source:         str,           # 'sensor' | 'weatherapi'
    api_response:   dict,
    recorded_at:    Optional[datetime] = None,
) -> dict:
    """
    Insert one row into sensor_readings.
    Returns the inserted row (with generated id and recorded_at).
    """
    row = {
        "recorded_at":    (recorded_at or datetime.now(timezone.utc)).isoformat(),
        "temperature":    temperature,
        "humidity":       humidity,
        "heat_index":     heat_index,
        "is_day":         is_day,
        "pressure_mb":    pressure_mb,
        "wind_kph":       wind_kph,
        "wind_gust_kph":  wind_gust_kph,
        "wind_degree":    wind_degree,
        "wind_dir_sin":   wind_dir_sin,
        "wind_dir_cos":   wind_dir_cos,
        "precip_mm":      precip_mm,
        "rain_mm":        rain_mm,
        "cloud_cover":    cloud_cover,
        "condition_text": condition_text,
        "source":         source,
        "api_response":   api_response,
    }
    result = _client.table("sensor_readings").insert(row).execute()
    return result.data[0]


# ---------------------------------------------------------------------------
# TABLE: hourly_data
# ---------------------------------------------------------------------------
def upsert_hourly_data(
    *,
    hour_start: datetime,
) -> Optional[dict]:
    """
    Aggregate sensor_readings for the given clock-hour and upsert into
    hourly_data. If no sensor rows exist, falls back to WeatherAPI values
    stored in the same hour's 'weatherapi' source rows.

    Args:
        hour_start: timezone-aware datetime truncated to the hour.

    Returns:
        The upserted hourly_data row, or None on error.
    """
    hour_end = hour_start + timedelta(hours=1)

    # Fetch all sensor_readings within this hour
    result = (
        _client.table("sensor_readings")
        .select("*")
        .gte("recorded_at", hour_start.isoformat())
        .lt("recorded_at", hour_end.isoformat())
        .execute()
    )
    rows = result.data

    sensor_rows = [r for r in rows if r["source"] == "sensor"]
    any_rows    = rows  # includes weatherapi fallback rows

    if not any_rows:
        # Nothing at all for this hour — skip
        return None

    def _avg(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _sum(lst, key):
        vals = [r[key] for r in lst if r.get(key) is not None]
        return round(sum(vals), 4) if vals else 0.0

    # Prefer sensor averages; fall back to all-rows averages
    use_rows = sensor_rows if sensor_rows else any_rows
    source   = "sensor_avg" if sensor_rows else "weatherapi_fallback"

    hs = hour_start
    row = {
        "hour_start":       hs.isoformat(),
        "hour_end":         hour_end.isoformat(),
        "avg_temperature":  _avg(use_rows, "temperature"),
        "avg_humidity":     _avg(use_rows, "humidity"),
        "avg_heat_index":   _avg(use_rows, "heat_index"),
        "avg_pressure_mb":  _avg(any_rows, "pressure_mb"),   # always use all rows
        "avg_wind_kph":     _avg(any_rows, "wind_kph"),
        "avg_wind_gust_kph":_avg(any_rows, "wind_gust_kph"),
        "total_precip_mm":  _sum(any_rows, "precip_mm"),
        "avg_cloud_cover":  _avg(any_rows, "cloud_cover"),
        "avg_wind_dir_sin": _avg(any_rows, "wind_dir_sin"),
        "avg_wind_dir_cos": _avg(any_rows, "wind_dir_cos"),
        # Temporal (derived from hour_start)
        "hour_of_day":      hs.hour,
        "day_of_week":      hs.weekday(),
        "month":            hs.month,
        "day_of_year":      hs.timetuple().tm_yday,
        "is_weekend":       hs.weekday() >= 5,
        "is_day":           1 if 6 <= hs.hour < 20 else 0,
        # Provenance
        "reading_count":    len(sensor_rows),
        "source":           source,
    }

    result = (
        _client.table("hourly_data")
        .upsert(row, on_conflict="hour_start")
        .execute()
    )
    return result.data[0] if result.data else None


def upsert_daily_data(*, date: datetime) -> Optional[dict]:
    """
    Aggregate hourly_data for the given calendar date and upsert into
    daily_data.

    Args:
        date: any datetime whose .date() identifies the target day.
    """
    day_start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    day_end   = day_start + timedelta(days=1)

    result = (
        _client.table("hourly_data")
        .select("*")
        .gte("hour_start", day_start.isoformat())
        .lt("hour_start", day_end.isoformat())
        .execute()
    )
    rows = result.data
    if not rows:
        return None

    def _avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _min(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(min(vals), 4) if vals else None

    def _max(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(max(vals), 4) if vals else None

    def _sum(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals), 4) if vals else 0.0

    sensor_hours   = sum(1 for r in rows if r["source"] == "sensor_avg")
    fallback_hours = sum(1 for r in rows if r["source"] == "weatherapi_fallback")

    row = {
        "date":             day_start.date().isoformat(),
        "min_temperature":  _min("avg_temperature"),
        "max_temperature":  _max("avg_temperature"),
        "avg_temperature":  _avg("avg_temperature"),
        "avg_humidity":     _avg("avg_humidity"),
        "avg_pressure_mb":  _avg("avg_pressure_mb"),
        "avg_wind_kph":     _avg("avg_wind_kph"),
        "total_precip_mm":  _sum("total_precip_mm"),
        "avg_cloud_cover":  _avg("avg_cloud_cover"),
        "day_of_week":      day_start.weekday(),
        "month":            day_start.month,
        "day_of_year":      day_start.timetuple().tm_yday,
        "is_weekend":       day_start.weekday() >= 5,
        "hours_with_data":  len(rows),
        "hours_sensor":     sensor_hours,
        "hours_fallback":   fallback_hours,
    }

    result = (
        _client.table("daily_data")
        .upsert(row, on_conflict="date")
        .execute()
    )
    return result.data[0] if result.data else None


# ---------------------------------------------------------------------------
# TABLE: model_performance
# ---------------------------------------------------------------------------
def insert_prediction(
    *,
    predicted_temp:    float,
    sensor_reading_id: int,
    feature_snapshot:  dict,
    model_version:     str = "sgd_v1",
    predicted_at:      Optional[datetime] = None,
) -> dict:
    """
    Log one prediction. actual_temp is NULL — back-filled on the next request
    via backfill_last_actual().
    """
    row = {
        "predicted_at":     (predicted_at or datetime.now(timezone.utc)).isoformat(),
        "predicted_temp":   round(predicted_temp, 4),
        "actual_temp":      None,
        "sensor_reading_id": sensor_reading_id,
        "feature_snapshot": feature_snapshot,
        "model_version":    model_version,
    }
    result = _client.table("model_performance").insert(row).execute()
    return result.data[0]


def backfill_last_actual(actual_temp: float) -> int:
    """
    Fill actual_temp on the most recent model_performance row where it is NULL.
    Called at the start of each /esp32-ingest request with the current sensor
    temperature — which is the ground truth for the *previous* prediction.

    Returns:
        Number of rows updated (0 or 1).
    """
    # Find the latest prediction without an actual value
    result = (
        _client.table("model_performance")
        .select("id")
        .is_("actual_temp", "null")
        .order("predicted_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return 0

    row_id = result.data[0]["id"]
    _client.table("model_performance").update(
        {"actual_temp": round(actual_temp, 4)}
    ).eq("id", row_id).execute()
    return 1


def fetch_recent_errors(n: int = 168) -> list[dict]:
    """
    Fetch the last `n` rows from model_performance where actual_temp is known.
    Used by the auto-retraining scheduler to compute rolling RMSE.

    Default n=168 = 7 days × 24 hours.
    """
    result = (
        _client.table("model_performance")
        .select("predicted_at, predicted_temp, actual_temp, absolute_error")
        .not_.is_("actual_temp", "null")
        .order("predicted_at", desc=True)
        .limit(n)
        .execute()
    )
    return result.data


# ---------------------------------------------------------------------------
# FIXED: fetch_recent_errors — data-relative, not calendar-relative
# ---------------------------------------------------------------------------
def fetch_recent_errors(n: int = 168) -> list:
    """
    Fetch the last `n` rows from model_performance where actual_temp is known.
    Orders by predicted_at DESC so we always get the most recent data
    regardless of whether it's from today or months ago (historical backfill).

    The old version used a NOW()-based filter which broke for historical data.
    """
    result = (
        _client.table("model_performance")
        .select("predicted_at, predicted_temp, actual_temp, absolute_error")
        .not_.is_("actual_temp", "null")
        .order("predicted_at", desc=True)
        .limit(n)
        .execute()
    )
    return result.data


def fetch_model_rmse_summary(days_back: int = 7) -> dict:
    """
    Call the model_rmse_summary(days_back) Postgres function.
    Returns overall RMSE/MAE for the most recent `days_back` days of data.
    Uses data-relative anchor (MAX predicted_at), not NOW().
    """
    result = _client.rpc("model_rmse_summary", {"days_back": days_back}).execute()
    return result.data[0] if result.data else {}


def fetch_hourly_trend(limit: int = 168) -> list:
    """Fetch rows from hourly_temp_trend view for frontend charts."""
    result = (
        _client.table("hourly_temp_trend")
        .select("*")
        .order("hour_start", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data


def fetch_daily_accuracy() -> list:
    """Fetch per-day MAE/RMSE from prediction_accuracy_daily view."""
    result = (
        _client.table("prediction_accuracy_daily")
        .select("*")
        .order("date", desc=True)
        .execute()
    )
    return result.data