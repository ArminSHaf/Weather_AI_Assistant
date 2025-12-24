import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

source_csv = "weather_romania_38_cities_2021_2025.csv"
output_csv = "training_6h_dataset.csv"

Past_hours = 24
future_Horizon = 6  # length of times that the model is going to predict

TIME_COL = "time"
CITY_COL = "city_id"

STEP_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover"
]  # dynamic and changes everytime

STATIC_FEATURES = [
    "latitude",
    "longitude",
    "elevation"
]  # static features remain the same

TARGET_FEATURES = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "weather_code",
]

CHUNK_SIZE = 10000  # controls RAM usage


# def map_wmo_to_condition(wmo: int) -> int:
#     if wmo in [0, 1, 2, 3]:
#         return 0
#     if 40 <= wmo <= 49:
#         return 1  # fog
#     if 51 <= wmo <= 67:
#         return 2  # rain
#     if 80 <= wmo <= 82:
#         return 2  # rain
#     if (71 <= wmo <= 77) or wmo in [85, 86]:
#         return 3  # snow / ice
#     if 95 <= wmo <= 99:
#         return 4  # thunderstorm
#     return 0


print("Loading CSV...")
df = pd.read_csv(source_csv)
print("loaded csv file")


df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = df.sort_values([CITY_COL, TIME_COL]).reset_index(drop=True)
# sort first by city then time - already have but garanteed.

df["hour"] = df[TIME_COL].dt.hour
df["dow"] = df[TIME_COL].dt.dayofyear


df["hour_sin"] = np.sin(2 * np.pi * df["hour"]/24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"]/24)

df["dow_sin"] = np.sin(2 * np.pi * df["dow"]/365)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"]/365)

TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

# verrrrry important for LSTM WE MUST USE SCALER TO SCALE ALL THE FEATURES

scaler = StandardScaler()
SCALE_COLS = STEP_FEATURES + STATIC_FEATURES
df[SCALE_COLS] = scaler.fit_transform(df[SCALE_COLS])
joblib.dump(scaler, "scaler.pkl")

# ONLY FOR NOT CRASHING SPLITING INTO CHUNKS
x_buffer, y_buffer = [], []
chunk_id = 0


def save_chunk(x_buf, y_buf, chunk_id):
    x_arr = np.stack(x_buf).astype(np.float32)
    y_arr = np.stack(y_buf).astype(np.float32)
    np.save(f"X_train_part_{chunk_id}.npy", x_arr)
    np.save(f"y_train_part_{chunk_id}.npy", y_arr)
    print(f"Saved chunk {chunk_id}")


for city_id, city_df in df.groupby(CITY_COL):
    city_df = city_df.reset_index(drop=True)

    n = len(city_df)

    # sampling
    for i in tqdm(range(Past_hours, n-future_Horizon), desc=f"city {city_id}", leave=False):

        past = city_df.iloc[i - Past_hours: i]

        step_mat = past[STEP_FEATURES].values
        time_mat = past[TIME_FEATURES].values

        static_vals = city_df.iloc[i][STATIC_FEATURES].values
        static_mat = np.repeat(static_vals[None, :], Past_hours, axis=0)
        # repeat it for 24 times  (24, 3) the same since static but must match the other datas

        X_seq = np.concatenate([step_mat, time_mat, static_mat], axis=1)

        future = city_df.iloc[i:i + future_Horizon][TARGET_FEATURES].values

        if np.any(pd.isna(future)):
            continue

        x_buffer.append(X_seq)
        y_buffer.append(future)

        if len(x_buffer) >= CHUNK_SIZE:
            save_chunk(x_buffer, y_buffer, chunk_id)
            x_buffer.clear()
            y_buffer.clear()
            chunk_id += 1


if x_buffer:
    save_chunk(x_buffer, y_buffer, chunk_id)

print("done!")
