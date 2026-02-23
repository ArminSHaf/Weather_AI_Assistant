# Weather Assistant – Short-Term Forecast Model

**Author:** Shafiei Armin - Turcu Mihai Alexandru - Scrob Sebastian  

## Description

This project builds a short-term weather prediction dataset for Romania’s main cities. It uses a public historical weather API (hourly) to download past weather for the last 4 years (temperature, humidity, pressure, wind, precipitation, cloud cover, weather code, elevation). The data is then merged into a single “mega” CSV file that will later be used to train machine learning models for 1–6 hour ahead forecasts per city.

## Setup

1. Clone this repository or download the project files.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 1 – Download city weather data

Run:

```bash
python retrieve_data.py
```
**make sure you have around 2GB storage at your directory**
- This script calls the historical weather API for each configured Romanian city 
- It saves one CSV file per city .

## Step 2 – Merge into a single dataset

After the per-city downloads are complete, run:

```bash
python merge_data.py
```
- this script will simply adds all csv files on top of one another and make the mega data set which later we can use for training the model.

## Step 3 – Build supervised training dataset

After the merged CSV is created (e.g. `weather_romania_38_cities_2021_2025.csv`), run:

```bash
python build_trainingset.py
```


This script:

- Sorts data by `city_id` and `time`, then builds **sliding windows** per city.
- Uses the **past 24 hours** of all core features as inputs:
  - all features plus,
- `hour`, `dayofyear`, 'solar_approx' (approximated) 
- Creates **targets** for horizons 1–6 hours ahead:
  - `target_wind_speed_h1..h6`
  - `target_temperature_h1..h6`
  - `target_precipitation_h1..h6`
  - `target_wmo_class_h1..h6`
- Saves the data by smaller chunks in npy files (x - input) and (y - target)



## Step 4 – Training a LSTM model

After the `.npy` chunks are created (e.g. `X_train_part_*.npy`, `y_train_part_*.npy`), run:


```bash
python training.py
```


This script:

- Streams chunked `.npy` files through a `tf.data.Dataset` so training fits in RAM.
- Uses a **LSTM encoder–decoder** to predict the next 6 hours for all targets. 
- Trains two heads simultaneously:
  - `regression`: temperature, precipitation, wind speed (MSE + MAE).
  - `classification`: WMO weather class for each future hour (sparse categorical cross-entropy + accuracy).
- Shuffles, batches, applies early stopping and learning‑rate scheduling, and saves the best model to `weather_lstm_6h_checkpoint.keras`.
- epochs is set to 30 but the model wills stop training when is decent trained after some iteration



## Step 5 – Prediction and agentic workflow

After training, run:

```bash
python app.py
```


This script:

- Loads the prediction.py file, following the trained model and scalers.  
- extract the city name from user query or default Bucharest.
- send the city name and start analysing using functions in `prediction.py`
- in prediction file :
  - Gets the selected city’s coordinates via the **Open‑Meteo API**.  
  - Downloads the last **24 hours** of data, computes extra features, and scales inputs.  
  - Runs `model.predict()` to generate 6‑hour forecasts for:
    - Temperature, wind, precipitation (regression).  
    - Weather condition (classification).  
  - Applies small corrections using the last measured values. 
- the agent will announce the predicted weather, suggesting the user about his/her cloths using gemini reasoning.
**note : the prediction is not quitely accurate** 



## step 6 - audio
Add your query as (audio) in `audio_folder/audio.wav` (e.g. “How is the weather in Brasov?”).
and then:
run the application :

```bash
python app.py
```

On Enter:
- The app transcribes `audio.wav` to text using `speech_recognition`.
- The transcribed text is sent to the weather agent, which generates a weather reply.  
- The reply text is converted to speech using *gTTS* and saved as `audio_folder/reply.wav`

