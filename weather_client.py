"""
weather_client.py
=================
WeatherAPI.com client — fetches current conditions and maps the response
fields to the exact keys expected by FeatureStateManager.

WeatherAPI current.json response fields used:
    current.humidity          -> humidity          (%)
    current.pressure_mb       -> pressure          (hPa / mb — same unit)
    current.wind_kph          -> wind_speed        (km/h)
    current.gust_kph          -> wind_gusts        (km/h)
    current.wind_degree       -> wind_degree       (° — converted to sin/cos)
    current.precip_mm         -> precipitation     (mm)
    current.precip_mm         -> rain              (mm, no snowfall in Nellore)
    current.cloud             -> cloud_cover       (%)
    current.is_day            -> is_day            (0/1)

Endpoint:
    GET https://api.weatherapi.com/v1/current.json
        ?key=<API_KEY>&q=<lat,lon>&aqi=no

Free tier: 1 million calls/month, no hourly restriction.
Sign up:   https://www.weatherapi.com/signup.aspx

Environment variables (loaded from .env):
    WEATHERAPI_KEY   — your WeatherAPI.com key (required)
    DEFAULT_LAT      — default latitude  (optional, falls back to Nellore)
    DEFAULT_LON      — default longitude (optional, falls back to Nellore)
"""

import math
import os

import httpx                          # pip install httpx
from dotenv import load_dotenv        # pip install python-dotenv

# ---------------------------------------------------------------------------
# LOAD .env  — must come before any os.getenv() calls
# ---------------------------------------------------------------------------
load_dotenv()                         # looks for .env in cwd, then parent dirs

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
WEATHERAPI_KEY      = os.getenv("WEATHERAPI_KEY")
WEATHERAPI_BASE_URL = "https://api.weatherapi.com/v1"

# Default location: Nellore, Andhra Pradesh (overridable via .env)
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "17.992"))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", "83.4251"))

if not WEATHERAPI_KEY:
    raise EnvironmentError(
        "WEATHERAPI_KEY is not set. "
        "Add it to your .env file or export it as an environment variable."
    )


# ---------------------------------------------------------------------------
# RESPONSE PARSER
# ---------------------------------------------------------------------------
def _parse_response(data: dict) -> dict:
    """
    Map WeatherAPI current.json response -> FeatureStateManager observation dict.

    WeatherAPI response structure:
        {
          "location": { "name": ..., "lat": ..., "lon": ... },
          "current": {
              "temp_c": 29.5,
              "humidity": 62,
              "pressure_mb": 1010.0,
              "wind_kph": 12.0,
              "gust_kph": 15.0,
              "wind_degree": 170,
              "precip_mm": 0.0,
              "cloud": 5,
              "is_day": 1,
              "feelslike_c": 33.2,
              ...
          }
        }
    """
    c = data["current"]

    wind_deg = float(c.get("wind_degree", 0))
    wind_rad = math.radians(wind_deg)

    return {
        # Used directly by FeatureStateManager
        "humidity":      float(c["humidity"]),
        "pressure":      float(c["pressure_mb"]),
        "wind_speed":    float(c["wind_kph"]),
        "wind_gusts":    float(c.get("gust_kph", c["wind_kph"])),
        "wind_degree":   wind_deg,
        "wind_dir_sin":  math.sin(wind_rad),
        "wind_dir_cos":  math.cos(wind_rad),
        "precipitation": float(c.get("precip_mm", 0.0)),
        "rain":          float(c.get("precip_mm", 0.0)),
        "cloud_cover":   float(c.get("cloud", 0)),
        "is_day":        int(c.get("is_day", 1)),
        # Extra fields for API response / logging (not used as ML features)
        "feelslike_c":   float(c.get("feelslike_c", 0.0)),
        "condition":     c.get("condition", {}).get("text", ""),
    }


# ---------------------------------------------------------------------------
# SYNC CLIENT  (used inside FastAPI endpoints via run_in_threadpool)
# ---------------------------------------------------------------------------
def fetch_current_weather(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    api_key: str = WEATHERAPI_KEY,
    timeout: float = 5.0,
) -> dict:
    """
    Fetch current weather from WeatherAPI.com and return a mapped observation
    dict ready for FeatureStateManager.update().

    Args:
        lat:     Latitude  (default: loaded from .env or Nellore fallback)
        lon:     Longitude (default: loaded from .env or Nellore fallback)
        api_key: WeatherAPI key (loaded from .env / WEATHERAPI_KEY env var)
        timeout: HTTP timeout in seconds

    Returns:
        Observation dict with keys: humidity, pressure, wind_speed,
        wind_gusts, wind_dir_sin, wind_dir_cos, precipitation, rain,
        cloud_cover, is_day, feelslike_c, condition

    Raises:
        httpx.HTTPError on network failure
        KeyError if the API response schema changes unexpectedly
    """
    url    = f"{WEATHERAPI_BASE_URL}/current.json"
    params = {"key": api_key, "q": f"{lat},{lon}", "aqi": "no"}

    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()

    return _parse_response(resp.json())


# ---------------------------------------------------------------------------
# ASYNC CLIENT  (optional — for async FastAPI endpoints)
# ---------------------------------------------------------------------------
async def fetch_current_weather_async(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    api_key: str = WEATHERAPI_KEY,
    timeout: float = 5.0,
) -> dict:
    """Async version of fetch_current_weather for use with async FastAPI routes."""
    url    = f"{WEATHERAPI_BASE_URL}/current.json"
    params = {"key": api_key, "q": f"{lat},{lon}", "aqi": "no"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()

    return _parse_response(resp.json())


# ---------------------------------------------------------------------------
# QUICK TEST  (run directly: python weather_client.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    print("[Test] Fetching current weather for Nellore ...")
    try:
        obs = fetch_current_weather()
        print(json.dumps(obs, indent=2))
    except Exception as e:
        print(f"[Error] {e}")