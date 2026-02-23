import time
from tqdm import tqdm
import requests
import pandas as pd

BUCHAREST = (44.4268, 26.1025)
IASI = (47.1622, 27.5889)
CLUJ_NAPOCA = (46.7667, 23.6000)
TIMISOARA = (45.7597, 21.2300)
CONSTANTA = (44.1800, 28.6500)
CRAIOVA = (44.3167, 23.8000)
BRASOV = (45.6500, 25.6000)
GALATI = (45.4500, 28.0500)
PLOIESTI = (44.9500, 26.0167)
ORADEA = (47.0722, 21.9211)
BRAILA = (45.2692, 27.9575)
ARAD = (46.1833, 21.3167)
PITESTI = (44.8667, 24.8833)
SIBIU = (45.7928, 24.1519)
BACAU = (46.5833, 26.9167)
TARGU_MURES = (46.5456, 24.5625)
BAIA_MARE = (47.6567, 23.5719)
BUZAU = (45.1531, 26.8208)
RAMNICU_VALCEA = (45.1047, 24.3756)
SATU_MARE = (47.7900, 22.8900)
BOTOSANI = (47.7486, 26.6694)
SUCEAVA = (47.6514, 26.2556)
RESITA = (45.3008, 21.8892)
DROBETA_TURNU_SEVERIN = (44.6333, 22.6500)
PIATRA_NEAMT = (46.9275, 26.3708)
BISTRITA = (47.1333, 24.5000)
TARGU_JIU = (45.0342, 23.2747)
TARGOVISTE = (44.9244, 25.4572)
FOCSANI = (45.7000, 27.1797)
TULCEA = (45.1900, 28.8000)
ALBA_IULIA = (46.0669, 23.5700)
SLATINA = (44.4297, 24.3642)
VASLUI = (46.6383, 27.7292)
CALARASI = (44.2000, 27.3333)
GIURGIU = (43.9008, 25.9739)
POPESTI_LEORDENI = (44.3800, 26.1700)
DEVA = (45.8781, 22.9144)
BARLAD = (46.2167, 27.6667)
ZALAU = (47.1911, 23.0572)
HUNEDOARA = (45.7697, 22.9203)
FLORESTI = (46.7475, 23.4908)
SFANTU_GHEORGHE = (45.8636, 25.7875)
ROMAN = (46.9300, 26.9300)
VOLUNTARI = (44.4925, 26.1914)
TURDA = (46.5667, 23.7833)
MIERCUREA_CIUC = (46.3594, 25.8017)
SLOBOZIA = (44.5639, 27.3661)
ALEXANDRIA = (43.9686, 25.3333)
BRAGADIRU = (44.3708, 25.9750)


CITIES = [BUCHAREST, IASI, CLUJ_NAPOCA, TIMISOARA, CONSTANTA, CRAIOVA, BRASOV, GALATI, PLOIESTI, ORADEA, BRAILA, ARAD, PITESTI, SIBIU, BACAU, TARGU_MURES, BAIA_MARE, BUZAU, RAMNICU_VALCEA, SATU_MARE, BOTOSANI, SUCEAVA, RESITA, DROBETA_TURNU_SEVERIN, PIATRA_NEAMT,
          BISTRITA, TARGU_JIU, TARGOVISTE, FOCSANI, TULCEA, ALBA_IULIA, SLATINA, VASLUI, CALARASI, GIURGIU, POPESTI_LEORDENI, DEVA, BARLAD, ZALAU, HUNEDOARA, FLORESTI, SFANTU_GHEORGHE, ROMAN, VOLUNTARI, TURDA, MIERCUREA_CIUC, SLOBOZIA, ALEXANDRIA, BRAGADIRU]


BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
TIMEZONE = "UTC"
# 2-3 hours earlier than local Romanian Time

YEARS = [
    ("2021-12-23", "2022-12-22"),
    ("2022-12-23", "2023-12-22"),
    ("2023-12-23", "2024-12-22"),
    ("2024-12-23", "2025-12-22"),
]


HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "cloud_cover",
    "weather_code",
]

# ----------------------------
# 3) Download one city (with elevation + retries)
# ----------------------------


def download_city_data(lat, lon, city_id, max_retries=3):
    dfs = []
    elevation_value = None

    for start_date, end_date in YEARS:
        for attempt in range(max_retries):
            try:
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start_date,
                    "end_date": end_date,
                    # list -> string (temperature_2m,relative_humidity_2m...)
                    "hourly": ",".join(HOURLY_VARS),
                    "timezone": TIMEZONE,
                }

                resp = requests.get(BASE_URL, params=params, timeout=(10, 180))
                # (connect time out , read time out)
                resp.raise_for_status()
                data = resp.json()

                if elevation_value is None:
                    elevation_value = data.get("elevation", None)

                if "hourly" not in data:
                    print(
                        f"[WARN] No hourly data for city {city_id}in {start_date} – {end_date}")
                    break

                hourly = data["hourly"]
                df = pd.DataFrame(hourly)
                df["latitude"] = lat
                df["longitude"] = lon
                df["city_id"] = city_id
                df["elevation"] = elevation_value
                dfs.append(df)
                break

            except requests.exceptions.ReadTimeout:
                print(
                    f"[TIMEOUT] city {city_id}, {start_date}–{end_date}, attempt {attempt+1}")
                time.sleep(5)
            except Exception as e:
                print(
                    f"[ERROR] city {city_id} , {start_date}–{end_date}: {e}")
                break

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


# START_INDEX = 37 - Had to many requests after city number 37


for idx, (lat, lon) in enumerate(tqdm(CITIES)):
    # if idx < START_INDEX:
    #     continue

    df_city = download_city_data(lat, lon, idx)
    if df_city is not None:
        df_city.to_csv(f"weather_ro_city_{idx}.csv", index=False)

    time.sleep(10)  # small pause to reduce rate-limit risk
