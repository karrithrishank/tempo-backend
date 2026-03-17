"""
scheduler.py
============
APScheduler background jobs for the temperature forecasting system.

Jobs:
    hourly_aggregation   — runs at :02 past every hour
                           aggregates sensor_readings -> hourly_data
                           if sensor offline: fetches WeatherAPI and inserts fallback

    daily_aggregation    — runs at 00:05 every day
                           aggregates previous day's hourly_data -> daily_data

    model_health_check   — runs every 6 hours
                           computes rolling RMSE from model_performance
                           triggers partial_fit retraining if RMSE > threshold

    weatherapi_offline   — runs every 10 minutes
                           if no sensor reading in last 10 min, inserts a
                           WeatherAPI fallback row into sensor_readings

Usage (started automatically by main.py lifespan):
    from scheduler import start_scheduler, stop_scheduler
    start_scheduler()
    stop_scheduler()
"""

import logging
import math
import numpy as np
import joblib
from datetime import datetime, timezone, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from db_client import (
    get_client,
    insert_sensor_reading,
    upsert_hourly_data,
    upsert_daily_data,
    fetch_recent_errors,
)
from weather_client import fetch_current_weather, DEFAULT_LAT, DEFAULT_LON
from utils import (
    INCREMENTAL_MODEL_PATH, SCALER_PATH,
    FEATURE_COLS, FeatureStateManager,
    predict_temperature, update_model_online,
)

log = logging.getLogger(__name__)

# RMSE threshold above which the model triggers a retraining pass
RMSE_RETRAIN_THRESHOLD = 1.5   # °C

_scheduler = BackgroundScheduler(timezone="UTC")


# ---------------------------------------------------------------------------
# JOB 1: WeatherAPI fallback — every 10 minutes
# ---------------------------------------------------------------------------
def job_weatherapi_fallback():
    """
    If the ESP32 hasn't posted in the last 10 minutes, insert a WeatherAPI
    row into sensor_readings so hourly aggregation always has data.
    """
    client = get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()

    result = (
        client.table("sensor_readings")
        .select("id")
        .eq("source", "sensor")
        .gte("recorded_at", cutoff)
        .limit(1)
        .execute()
    )

    if result.data:
        return  # sensor is alive — nothing to do

    log.info("[Scheduler] ESP32 offline — inserting WeatherAPI fallback row")
    try:
        meteo = fetch_current_weather(DEFAULT_LAT, DEFAULT_LON)
    except Exception as exc:
        log.error(f"[Scheduler] WeatherAPI fetch failed: {exc}")
        return

    wind_deg = meteo.get("wind_degree", 0.0)
    insert_sensor_reading(
        temperature=    None,               # ESP32 offline — no sensor temp
        humidity=       meteo["humidity"],
        heat_index=     None,
        is_day=         meteo["is_day"],
        pressure_mb=    meteo["pressure"],
        wind_kph=       meteo["wind_speed"],
        wind_gust_kph=  meteo["wind_gusts"],
        wind_degree=    wind_deg,
        wind_dir_sin=   meteo["wind_dir_sin"],
        wind_dir_cos=   meteo["wind_dir_cos"],
        precip_mm=      meteo["precipitation"],
        rain_mm=        meteo["rain"],
        cloud_cover=    meteo["cloud_cover"],
        condition_text= meteo.get("condition", ""),
        source=         "weatherapi",
        api_response=   meteo,
    )


# ---------------------------------------------------------------------------
# JOB 2: Hourly aggregation — at HH:02 every hour
# ---------------------------------------------------------------------------
def job_hourly_aggregation():
    """
    Aggregate the previous clock-hour's sensor_readings into hourly_data.
    Runs at :02 past every hour to allow the last few sensor posts to land.
    """
    now        = datetime.now(timezone.utc)
    prev_hour  = (now - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    log.info(f"[Scheduler] Aggregating hourly data for {prev_hour.isoformat()}")
    try:
        row = upsert_hourly_data(hour_start=prev_hour)
        if row:
            log.info(f"[Scheduler] hourly_data upserted: source={row['source']}, "
                     f"readings={row['reading_count']}")
        else:
            log.warning(f"[Scheduler] No data for hour {prev_hour.isoformat()} — skipped")
    except Exception as exc:
        log.error(f"[Scheduler] Hourly aggregation failed: {exc}")


# ---------------------------------------------------------------------------
# JOB 3: Daily aggregation — at 00:05 every day
# ---------------------------------------------------------------------------
def job_daily_aggregation():
    """
    Aggregate the previous calendar day's hourly_data into daily_data.
    """
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    log.info(f"[Scheduler] Aggregating daily data for {yesterday.date()}")
    try:
        row = upsert_daily_data(date=yesterday)
        if row:
            log.info(f"[Scheduler] daily_data upserted for {yesterday.date()}: "
                     f"sensor_hours={row['hours_sensor']}, "
                     f"fallback_hours={row['hours_fallback']}")
        else:
            log.warning(f"[Scheduler] No hourly data for {yesterday.date()} — skipped")
    except Exception as exc:
        log.error(f"[Scheduler] Daily aggregation failed: {exc}")


# ---------------------------------------------------------------------------
# JOB 4: Model health check + auto-retrain — every 6 hours
# ---------------------------------------------------------------------------
def job_model_health_check():
    """
    1. Fetch last 168 model_performance rows (7 days) from Supabase.
    2. Compute rolling RMSE and MAE.
    3. If RMSE > RMSE_RETRAIN_THRESHOLD, run a partial_fit pass over the
       error rows to nudge the model back towards recent distribution.
    4. Log results.
    """
    log.info("[Scheduler] Running model health check ...")
    try:
        rows = fetch_recent_errors(n=168)
    except Exception as exc:
        log.error(f"[Scheduler] fetch_recent_errors failed: {exc}")
        return

    if len(rows) < 10:
        log.info("[Scheduler] Not enough predictions yet for health check.")
        return

    errors = [r["absolute_error"] for r in rows if r["absolute_error"] is not None]
    if not errors:
        return

    mae  = round(sum(errors) / len(errors), 4)
    rmse = round(math.sqrt(sum(e**2 for e in errors) / len(errors)), 4)
    log.info(f"[Scheduler] Model health — MAE={mae}°C  RMSE={rmse}°C  "
             f"n={len(errors)} predictions")

    if rmse <= RMSE_RETRAIN_THRESHOLD:
        log.info("[Scheduler] RMSE within threshold — no retraining needed.")
        return

    # Retrain: partial_fit on the recent error rows
    log.warning(f"[Scheduler] RMSE {rmse} > threshold {RMSE_RETRAIN_THRESHOLD} "
                "— triggering incremental retraining ...")
    try:
        model  = joblib.load(INCREMENTAL_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        fsm    = FeatureStateManager.load()

        retrain_count = 0
        for row in reversed(rows):   # oldest first
            if row.get("actual_temp") is None:
                continue
            # Re-use the stored feature snapshot
            snap = row.get("feature_snapshot")
            if not snap:
                continue
            x = np.array([snap.get(c, 0.0) for c in FEATURE_COLS], dtype=float)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
            model.partial_fit(scaler.transform(x), [row["actual_temp"]])
            retrain_count += 1

        joblib.dump(model, INCREMENTAL_MODEL_PATH)
        log.info(f"[Scheduler] Retraining done — {retrain_count} samples used.")
    except Exception as exc:
        log.error(f"[Scheduler] Retraining failed: {exc}")


# ---------------------------------------------------------------------------
# SCHEDULER LIFECYCLE
# ---------------------------------------------------------------------------
def start_scheduler():
    """Register all jobs and start the background scheduler."""
    # Job 1: WeatherAPI offline fallback — every 10 minutes
    _scheduler.add_job(
        job_weatherapi_fallback,
        trigger=IntervalTrigger(minutes=10),
        id="weatherapi_fallback",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=60,
    )

    # Job 2: Hourly aggregation — at HH:02
    _scheduler.add_job(
        job_hourly_aggregation,
        trigger=CronTrigger(minute=2),
        id="hourly_aggregation",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )

    # Job 3: Daily aggregation — at 00:05 UTC
    _scheduler.add_job(
        job_daily_aggregation,
        trigger=CronTrigger(hour=0, minute=5),
        id="daily_aggregation",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=600,
    )

    # Job 4: Model health check — every 6 hours
    _scheduler.add_job(
        job_model_health_check,
        trigger=CronTrigger(hour="0,6,12,18", minute=10),
        id="model_health_check",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=600,
    )

    _scheduler.start()
    log.info("[Scheduler] All jobs registered and scheduler started.")


def stop_scheduler():
    """Gracefully shut down the scheduler."""
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        log.info("[Scheduler] Scheduler stopped.")