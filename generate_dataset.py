import requests
import pandas as pd
from datetime import datetime

# -------------------------------
# Configuration
# -------------------------------
START_YEAR = 2019
END_YEAR = 2024
IDI_THRESHOLD = 30  # irrigation threshold

regions = {
    "Mandya": (12.5242, 76.8958),
    "Raichur": (16.2076, 77.3463),
    "Tumakuru": (13.3392, 77.1173),
    "Anantapur": (14.6819, 77.6006),
    "Warangal": (17.9689, 79.5941),
}

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


# -------------------------------
# Fetch NASA POWER Data
# -------------------------------
def fetch_region_data(region, lat, lon, start, end):
    params = {
        "parameters": "PRECTOTCORR,EVPTRNS",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON",
    }

    response = requests.get(NASA_URL, params=params, timeout=60)
    data = response.json()

    if "properties" not in data:
        raise RuntimeError(f"NASA API failed for {region}: {data}")

    rainfall = data["properties"]["parameter"]["PRECTOTCORR"]
    eto = data["properties"]["parameter"]["EVPTRNS"]

    df = pd.DataFrame({
        "date": rainfall.keys(),
        "rainfall": rainfall.values(),
        "eto": eto.values(),
    })

    df["date"] = pd.to_datetime(df["date"])
    df["region"] = region

    return df


# -------------------------------
# Compute IDI
# -------------------------------
def compute_idi(df):
    df = df.sort_values("date").copy()
    df["idi_daily"] = df["eto"] - df["rainfall"]
    df["idi_7day"] = df["idi_daily"].rolling(7).sum()
    df.dropna(inplace=True)
    return df


# -------------------------------
# MAIN DATA GENERATION
# -------------------------------
all_data = []

for region, (lat, lon) in regions.items():
    print(f"Fetching data for {region}...")

    start = f"{START_YEAR}0101"
    end = f"{END_YEAR}1231"

    df = fetch_region_data(region, lat, lon, start, end)
    df = compute_idi(df)

    # Auto-label using rule-based logic
    df["action_needed"] = (df["idi_7day"] > IDI_THRESHOLD).astype(int)

    all_data.append(df)

# Combine all regions
final_df = pd.concat(all_data, ignore_index=True)

# Save dataset
final_df.to_csv("nasa_irrigation_dataset.csv", index=False)

print("\n✅ Dataset generation complete!")
print("Saved as: nasa_irrigation_dataset.csv")
print("Total samples:", len(final_df))
print(final_df["action_needed"].value_counts())
