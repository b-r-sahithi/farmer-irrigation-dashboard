import requests
import pandas as pd

def fetch_nasa_power(lat, lon, start, end):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "PRECTOTCORR,EVPTRNS",   # ✅ FIXED HERE
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON"
    }

    response = requests.get(url, params=params, timeout=30)
    data = response.json()

    # Safety check
    if "properties" not in data:
        raise RuntimeError(f"NASA POWER API error: {data}")

    params_data = data["properties"]["parameter"]

    rainfall = params_data["PRECTOTCORR"]
    eto = params_data["EVPTRNS"]

    df = pd.DataFrame({
        "date": rainfall.keys(),
        "rainfall": rainfall.values(),
        "eto": eto.values()
    })

    df["date"] = pd.to_datetime(df["date"])
    return df
