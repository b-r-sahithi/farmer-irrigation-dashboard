import pandas as pd

def compute_idi(df):
    df = df.sort_values("date")
    df["rainfall"] = df["rainfall"].fillna(0)
    df["eto"] = df["eto"].fillna(df["eto"].mean())

    # Irrigation Deficit Index
    df["idi"] = df["eto"] - df["rainfall"]
    df["idi_7day"] = df["idi"].rolling(7).sum()

    df.dropna(inplace=True)
    return df
