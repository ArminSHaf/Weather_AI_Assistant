import glob
import pandas as pd

files = sorted(glob.glob("weather_ro_city_*.csv"))
# return all the files in the directory with the format provided

print(f"found {len(files)} files")

dfs = []

for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
# stack data frames on top of each other -> concat
# ignore_index = reset rows to 0 - avoid duplication

df_all.to_csv("weather_romania_38_cities_2021_2025.csv", index=False)

print("Merged shape : ", df_all.shape)
