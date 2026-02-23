import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import requests
from datetime import datetime, timedelta

MODEL_PATH = "weather_lstm_6h_prediction.keras"
SCALER_MAIN = "scaler.pkl"
SCALER_TEMP = "scaler_temp.pkl"
SCALER_PRECIP = "scaler_precip.pkl"
SCALER_WIND = "scaler_wind.pkl"

PAST_HOURS = 24
FUTURE_HORIZON = 6

STEP_FEATURES = [
    "temperature_2m",
    "temp_diff",            # Calculated
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "solar_approx"          # Calculated
]

CURRENT_LAT = 0.0
CURRENT_LON = 0.0
CURRENT_ELEV = 0.0

STATIC_FEATURES = ["latitude", "longitude", "elevation"]


def get_data_city(city):
    base_url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"cannot get geo correlations for the city provided: {e}")
        return None
    if "results" not in data or len(data["results"]) == 0:
        print("No results found for this ", city)
        return None, " no result found ..."

    result = data["results"][0]

    lat = result.get('latitude')
    lon = result.get('longitude')
    elev = result.get('elevation')
    print(
        f"got the {city}'s correlations. now searching for its weather nalysis for past 24h")
    return (lat, lon, elev), None


def get_live_data():

    api_cols = [
        "temperature_2m", "relative_humidity_2m", "surface_pressure",
        "wind_speed_10m", "wind_direction_10m", "precipitation", "cloud_cover"
    ]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": CURRENT_LAT,
        "longitude": CURRENT_LON,
        "hourly": ",".join(api_cols),
        "past_days": 2,
        "forecast_days": 1,
        "timezone": "UTC",
        "wind_speed_unit": "kmh"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return None, "weather API error"

    df = pd.DataFrame(data['hourly'])
    df['time'] = pd.to_datetime(df['time'])

    df['latitude'] = CURRENT_LAT
    df['longitude'] = CURRENT_LON
    df['elevation'] = CURRENT_ELEV

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    df_hist = df[df['time'] < now].tail(PAST_HOURS).copy()

    if len(df_hist) < 24:
        print(f"Error: Not enough data. Needed 24 rows, got {len(df_hist)}")
        return None, " not enough data"

    # print(df_hist.tail(24))
    return df_hist.reset_index(drop=True), None


def preprocess_data(df, scaler):
    # 1. CALCULATE PHYSICS FEATURES (V2 Logic)

    df['temp_diff'] = df['temperature_2m'].diff().fillna(0)

    # B. Synthetic Solar
    df["hour"] = df['time'].dt.hour
    # Sun angle math
    sun_angle = np.sin((df["hour"] - 6) * np.pi / 12)
    sun_angle = np.maximum(sun_angle, 0)
    # Cloud factor
    cloud_factor = 1 - (df['cloud_cover'] / 100.0)
    df['solar_approx'] = sun_angle * cloud_factor

    # ==========================================

    # 2. Time
    df["dow"] = df['time'].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"]/24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"]/24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"]/365)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"]/365)

    # 3. Scaling
    scale_cols = STEP_FEATURES + STATIC_FEATURES
    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.transform(df[scale_cols])

    # 4. Matrix Construction
    step_mat = df_scaled[STEP_FEATURES].values
    time_mat = df_scaled[["hour_sin", "hour_cos", "dow_sin", "dow_cos"]].values
    static_mat = df_scaled[STATIC_FEATURES].values

    X_seq = np.concatenate([step_mat, time_mat, static_mat], axis=1)

    last_raw_values = {
        "temp": float(df['temperature_2m'].iloc[-1]),
        "wind": float(df['wind_speed_10m'].iloc[-1])
    }

    return X_seq.reshape(1, PAST_HOURS, 16), df['time'].iloc[-1], last_raw_values


def decode_weather_smart(probs):
    """
    Look at the probabilities. If Rain/Snow has > 25% chance, report it.
    wmo code map
    """
    rain_prob = probs[3] + probs[4]
    snow_prob = probs[5]
    storm_prob = probs[6]

    if storm_prob > 0.20:
        return "Thunderstorm (Risk)"
    if snow_prob > 0.25:
        return "Snow (Possible)"
    if rain_prob > 0.25:
        return "Rain (Possible)"

    idx = np.argmax(probs)
    mapping = {
        0: "Clear", 1: "Cloudy", 2: "Fog",
        3: "Rain (Light)", 4: "Rain (Heavy)",
        5: "Snow/Ice", 6: "Thunderstorm"
    }
    return mapping.get(idx, "Unknown")


def predict_weather(city: str) -> dict:
    global CURRENT_LAT, CURRENT_LON, CURRENT_ELEV

    print("WEATHER PREDICTOR :")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_MAIN)
        t_scaler = joblib.load(SCALER_TEMP)
        p_scaler = joblib.load(SCALER_PRECIP)
        w_scaler = joblib.load(SCALER_WIND)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    coords, error = get_data_city(city)
    if error:
        return {"status": "error", "message": error}

    CURRENT_LAT, CURRENT_LON, CURRENT_ELEV = coords

    df, error = get_live_data()
    if error:
        return {"status": "error", "message": error}

    try:
        X_input, last_ts, last_raw = preprocess_data(df, scaler)
    except Exception as e:
        return {"status": "error", "message": f"Preprocessing error: {e}"}

    print("Predicting...")
    preds = model.predict(X_input, verbose=0)

    reg_pred = preds[0][0]
    cls_pred = preds[1][0]

    # Unscale predictions
    pred_temp = t_scaler.inverse_transform(
        reg_pred[:, 0].reshape(-1, 1)).flatten()
    pred_precip = p_scaler.inverse_transform(
        reg_pred[:, 1].reshape(-1, 1)).flatten()
    pred_wind = w_scaler.inverse_transform(
        reg_pred[:, 2].reshape(-1, 1)).flatten()

    # === DUAL ANCHORING ===    for better prediction avoid giving an unrelavant result than last hour

    temp_offset = (last_raw['temp'] - pred_temp[0]) * 0.9
    wind_offset = (last_raw['wind'] - pred_wind[0]) * 0.8

    forecast_data = []

    for i in range(FUTURE_HORIZON):
        corrected_temp = pred_temp[i] + temp_offset
        corrected_wind = max(0.0, pred_wind[i] + wind_offset)
        precip = max(0.0, pred_precip[i])
        cond = decode_weather_smart(cls_pred[i])

        forecast_data.append({
            "hour": i + 1,
            "temp_c": float(round(corrected_temp, 1)),
            "wind_kmh": float(round(corrected_wind, 1)),
            "precip_mm": float(round(precip, 2)),
            "condition": cond
        })

    return {
        "status": "success",
        "city_name": city,
        "current_observation": {
            "temp": last_raw['temp'],
            "wind": last_raw['wind']
        },
        "forecast": forecast_data
    }
