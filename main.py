"""
main.py
=======
FastAPI application — ESP32 temperature forecasting service.

Data flow per /esp32-ingest request:
    ESP32 POST
      -> backfill previous prediction's actual_temp in model_performance
      -> fetch WeatherAPI.com atmospheric context
      -> insert merged row into sensor_readings (source='sensor')
      -> update FeatureStateManager rolling buffer
      -> predict next-step temperature
      -> insert prediction into model_performance
      -> partial_fit SGD model with current true temperature
      -> persist model + feature state

Endpoints:
    POST /esp32-data          original passthrough (no ML, no DB)
    POST /esp32-ingest        full pipeline
    GET  /model/status        health check + rolling RMSE
    GET  /model/performance   last N predictions from Supabase

Schedulers (started on app startup):
    Every 10 min   WeatherAPI fallback row when ESP32 offline
    Every hour     Aggregate sensor_readings -> hourly_data
    Every day      Aggregate hourly_data -> daily_data
    Every 6 hours  Model RMSE health check + auto-retrain if needed

Environment variables (set in .env):
    WEATHERAPI_KEY    weatherapi.com key
    SUPABASE_URL      https://xxxx.supabase.co
    SUPABASE_KEY      service-role key
    WEATHER_LAT       sensor latitude  (default: 17.992 Nellore)
    WEATHER_LON       sensor longitude (default: 83.4251 Nellore)

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import logging
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

load_dotenv()

from utils import (
    FEATURE_COLS, INCREMENTAL_MODEL_PATH, SCALER_PATH,
    FeatureStateManager, predict_temperature, update_model_online,
)
from weather_client import fetch_current_weather, DEFAULT_LAT, DEFAULT_LON
from db_client import (
    insert_sensor_reading,
    insert_prediction,
    backfill_last_actual,
    fetch_recent_errors,
    fetch_model_rmse_summary,
    fetch_hourly_trend,
    fetch_daily_accuracy,
)
from scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SENSOR_LAT = float(os.getenv("WEATHER_LAT", DEFAULT_LAT))
SENSOR_LON = float(os.getenv("WEATHER_LON", DEFAULT_LON))


# ---------------------------------------------------------------------------
# LIFESPAN  — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, feature_state

    # Load ML artefacts
    try:
        model         = joblib.load(INCREMENTAL_MODEL_PATH)
        scaler        = joblib.load(SCALER_PATH)
        feature_state = FeatureStateManager.load()
        log.info("[Startup] Model loaded.")
    except Exception as exc:
        log.warning(f"[Startup] Models not loaded: {exc}. Run train.py first.")
        model = scaler = feature_state = None

    # Start background schedulers
    start_scheduler()
    log.info("[Startup] Schedulers started.")

    yield  # app runs here

    stop_scheduler()
    log.info("[Shutdown] Schedulers stopped.")


app = FastAPI(title="ESP32 Temperature Forecast API", lifespan=lifespan)

# Module-level references updated in lifespan
model:         object = None
scaler:        object = None
feature_state: object = None


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
class SensorData(BaseModel):
    """
    Payload from ESP32.
    light is now boolean (0/1 — PIR/LDR digital output).
    """
    temperature: float          # °C
    humidity:    float          # %
    heatIndex:   float          # °C
    light:       int            # 0 = dark, 1 = light detected


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.post("/esp32-data")
def receive_data(data: SensorData):
    """Original passthrough — no ML, no DB writes."""
    return {
        "status":      "received",
        "temperature": data.temperature,
        "humidity":    data.humidity,
        "heatIndex":   data.heatIndex,
        "light":       data.light,
    }


@app.post("/esp32-ingest")
async def ingest_and_predict(data: SensorData):
    """
    Full pipeline:
    1. Back-fill previous prediction's actual_temp with this reading
    2. Fetch WeatherAPI.com context
    3. Persist sensor + meteo to sensor_readings (source='sensor')
    4. Compute lag/rolling features
    5. Predict next-step temperature
    6. Log prediction to model_performance
    7. Incremental model update (partial_fit)
    8. Persist feature state
    """
    if model is None:
        raise HTTPException(503, "Model not loaded. Run python train.py first.")

    # Step 1 — back-fill previous prediction's actual_temp
    await run_in_threadpool(backfill_last_actual, data.temperature)

    # Step 2 — fetch WeatherAPI context
    try:
        meteo = await run_in_threadpool(fetch_current_weather, SENSOR_LAT, SENSOR_LON)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"WeatherAPI fetch failed: {exc}")

    # Step 3 — persist to DB
    wind_deg = meteo.get("wind_degree", 0.0)
    try:
        db_row = await run_in_threadpool(
            insert_sensor_reading,
            temperature=    data.temperature,
            humidity=       data.humidity,
            heat_index=     data.heatIndex,
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
            source=         "sensor",
            api_response=   meteo,
        )
        sensor_reading_id = db_row["id"]
    except Exception as exc:
        log.error(f"DB insert failed: {exc}")
        sensor_reading_id = None

    # Step 4 — build observation + enrich with lag/rolling features
    obs = {
        "temperature":   data.temperature,
        "humidity":      data.humidity,
        "pressure":      meteo["pressure"],
        "wind_speed":    meteo["wind_speed"],
        "wind_gusts":    meteo["wind_gusts"],
        "wind_dir_sin":  meteo["wind_dir_sin"],
        "wind_dir_cos":  meteo["wind_dir_cos"],
        "precipitation": meteo["precipitation"],
        "rain":          meteo["rain"],
        "cloud_cover":   meteo["cloud_cover"],
        "is_day":        meteo["is_day"],
    }
    feature_dict = feature_state.update(obs)

    # Step 5 — predict
    predicted_temp = predict_temperature(feature_dict, scaler, model)

    # Step 6 — log prediction
    try:
        await run_in_threadpool(
            insert_prediction,
            predicted_temp=     predicted_temp,
            sensor_reading_id=  sensor_reading_id,
            feature_snapshot=   feature_dict,
            model_version=      "sgd_v1",
        )
    except Exception as exc:
        log.error(f"Prediction insert failed: {exc}")

    # Step 7 — incremental update
    update_model_online(feature_dict, data.temperature, scaler, model)

    # Step 8 — persist feature state
    feature_state.save()

    return {
        "status":           "received",
        "current_temp_c":   data.temperature,
        "predicted_next_c": round(predicted_temp, 2),
        "humidity":         data.humidity,
        "heatIndex":        data.heatIndex,
        "light":            data.light,
        "weather_context": {
            "pressure_mb":      meteo["pressure"],
            "wind_kph":         meteo["wind_speed"],
            "gust_kph":         meteo["wind_gusts"],
            "precipitation_mm": meteo["precipitation"],
            "cloud_cover_pct":  meteo["cloud_cover"],
            "condition":        meteo.get("condition", ""),
            "is_day":           meteo["is_day"],
        },
    }


@app.get("/model/status")
async def model_status():
    """
    Health check + model performance stats.
    Uses data-relative RMSE (anchored to latest record, not NOW())
    so historical backfill data is always reflected correctly.
    """
    perf = {}
    try:
        # Uses MAX(predicted_at) as anchor — works for both live and historical data
        perf = await run_in_threadpool(fetch_model_rmse_summary, 7)
    except Exception as exc:
        log.warning(f"Could not fetch model RMSE: {exc}")

    return {
        "model_loaded":    model is not None,
        "model_type":      type(model).__name__ if model else "not loaded",
        "feature_count":   len(FEATURE_COLS),
        "history_length":  len(feature_state.history) if feature_state else 0,
        "sensor_location": {"lat": SENSOR_LAT, "lon": SENSOR_LON},
        "weather_source":  "weatherapi.com",
        # Performance over most recent 7 days of data (data-relative, not calendar)
        "rolling_rmse_7d": perf.get("rmse"),
        "rolling_mae_7d":  perf.get("mae"),
        "n_predictions":   perf.get("n_predictions"),
        "period_start":    perf.get("period_start"),
        "period_end":      perf.get("period_end"),
    }


@app.get("/model/performance")
async def model_performance(n: int = Query(default=48, ge=1, le=500)):
    """Return last N predictions with actuals from Supabase."""
    try:
        rows = await run_in_threadpool(fetch_recent_errors, n)
        return {"count": len(rows), "predictions": rows}
    except Exception as exc:
        raise HTTPException(500, f"DB query failed: {exc}")


@app.get("/analytics/hourly")
async def analytics_hourly(limit: int = Query(default=168, ge=1, le=720)):
    """
    Hourly temperature trend with predictions for frontend charts.
    Returns actual vs predicted temp, humidity, pressure, wind, precipitation.
    """
    try:
        rows = await run_in_threadpool(fetch_hourly_trend, limit)
        return {"count": len(rows), "data": rows}
    except Exception as exc:
        raise HTTPException(500, f"DB query failed: {exc}")


@app.get("/analytics/daily")
async def analytics_daily():
    """
    Daily aggregates + model accuracy per day for frontend dashboard.
    Returns min/max/avg temp, total precip, MAE, RMSE per day.
    """
    try:
        accuracy = await run_in_threadpool(fetch_daily_accuracy)
        return {"count": len(accuracy), "data": accuracy}
    except Exception as exc:
        raise HTTPException(500, f"DB query failed: {exc}")