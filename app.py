import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

from nasa_power import fetch_nasa_power
from preprocess import compute_idi

def estimate_ndvi_from_idi(df):
    """
    Estimate NDVI dynamically from recent irrigation stress.
    This is a proxy NDVI based on environmental conditions.
    """
    recent_idi = df["idi_7day"].tail(7).mean()

    NDVI_MAX = 0.75
    IDI_THRESHOLD = 30
    K = 0.4  # sensitivity factor

    ndvi = NDVI_MAX - K * (recent_idi / IDI_THRESHOLD)

    # Clamp NDVI to realistic range
    ndvi = max(0.3, min(0.75, ndvi))
    return ndvi

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Farmer Irrigation Advisory",
    layout="wide"
)

# ==================================================
# FARMER-FRIENDLY BIG UI STYLES
# ==================================================
st.markdown("""
<style>

/* GLOBAL FONT */
html, body, [class*="css"] {
    font-size: 20px !important;
}

/* MAIN TITLE */
.main-title {
    font-size: 46px !important;
    font-weight: 800;
    color: #1B5E20;
    text-align: center;
}

/* SUB HEADINGS */
h2, h3 {
    font-size: 34px !important;
    color: #2E7D32;
}

/* METRICS */
[data-testid="stMetricValue"] {
    font-size: 40px !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    font-size: 22px !important;
}

/* ALERTS */
.stAlert {
    font-size: 28px !important;
    font-weight: 700;
}

/* BUTTON */
button {
    font-size: 24px !important;
    padding: 12px 30px !important;
}

/* CAPTION */
.stCaption {
    font-size: 18px !important;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================
st.markdown(
    "<div class='main-title'>🌱 Farmer Irrigation Advisory System</div>",
    unsafe_allow_html=True
)
st.caption(
    "Simple guidance for farmers using satellite weather data"
)

# ==================================================
# LOAD ML MODEL (SUPPORT ONLY)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

# ==================================================
# SEASON FUNCTION
# ==================================================
def get_season(month):
    if month in [12, 1, 2]:
        return "❄️ Winter (Rabi Season)"
    elif month in [3, 4, 5]:
        return "🔥 Summer (Pre-Monsoon)"
    elif month in [6, 7, 8, 9]:
        return "🌧️ Monsoon (Kharif Season)"
    else:
        return "🍂 Post-Monsoon"

# ==================================================
# FARM LOCATION
# ==================================================
st.markdown("---")
st.subheader("📍 Select Your Farm Location")

mode = st.radio(
    "How do you want to select your farm?",
    ["Select Farming Region", "Enter Location Manually"],
    horizontal=True
)

regions = {
    "Mandya – Irrigated Area": (12.5242, 76.8958),
    "Raichur – Dryland Area": (16.2076, 77.3463),
    "Tumakuru – Mixed Crops": (13.3392, 77.1173),
    "Anantapur – Dry Region": (14.6819, 77.6006),
    "Warangal – Cotton Area": (17.9689, 79.5941),
}

if mode == "Select Farming Region":
    region = st.selectbox("Choose Region", list(regions.keys()))
    lat, lon = regions[region]
    st.success(f"Farm Selected → Latitude {lat:.4f}, Longitude {lon:.4f}")
else:
    lat = st.number_input("Latitude", -90.0, 90.0, 13.0)
    lon = st.number_input("Longitude", -180.0, 180.0, 77.0)

st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=7)

# ==================================================
# DATE RANGE
# ==================================================
st.markdown("---")
st.subheader("📅 Select Days to Check")

start_date = st.date_input("From Date")
end_date = st.date_input("To Date")

# ==================================================
# ANALYZE BUTTON
# ==================================================
st.markdown("---")
if st.button("🔍 CHECK MY FARM"):

    if start_date >= end_date:
        st.error("Start date must be before end date")
        st.stop()

    if (end_date - start_date).days < 10:
        st.warning("Please select at least 10 days")
        st.stop()

    df = fetch_nasa_power(
        lat, lon,
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d")
    )
    df = compute_idi(df)

    if df.empty:
        st.warning("Not enough data")
        st.stop()

    latest = df.iloc[-1]
    idi = latest["idi_7day"]

    # ==================================================
    # SEASON INFO
    # ==================================================
    season = get_season(latest["date"].month)
    st.info(f"🌦️ Current Season: **{season}**")

    # ==================================================
    # FARMER DECISION LOGIC
    # ==================================================
    if idi <= 0:
        alert = "🟢 NO IRRIGATION NEEDED"
        explain = "Rainfall is good. Soil has enough water."
        risk = 0
        color = "success"

    elif idi < 15:
        alert = "🟢 NO IRRIGATION NEEDED"
        explain = "Soil moisture is sufficient."
        risk = (idi / 30) * 100
        color = "success"

    elif idi < 30:
        alert = "🟡 WATCH YOUR FIELD"
        explain = "Soil moisture is reducing. Be ready to irrigate."
        risk = (idi / 30) * 100
        color = "warning"

    else:
        alert = "🔴 IRRIGATE YOUR CROP NOW"
        explain = "High water stress detected."
        risk = 100
        color = "error"

    risk = max(0, min(100, risk))

    # ==================================================
    # MAIN MESSAGE
    # ==================================================
    st.markdown("---")
    st.subheader("🚜 What should you do today?")

    if color == "success":
        st.success(alert)
    elif color == "warning":
        st.warning(alert)
    else:
        st.error(alert)

    st.markdown(
        f"<div style='font-size:26px; margin-top:10px;'>{explain}</div>",
        unsafe_allow_html=True
    )

    # ==================================================
    # KEY NUMBERS (BIG & SIMPLE)
    # ==================================================
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    col1.metric("💧 Water Stress", f"{idi:.1f}")
    col2.metric("⚠️ Irrigation Risk", f"{risk:.0f}%")

    ml_risk = model.predict_proba(
        [[latest["rainfall"], latest["eto"]]]
    )[0][1] * 100

    col3.metric("📊 Climate Risk", f"{ml_risk:.0f}%")

    # ==================================================
    # NDVI – REAL-TIME CROP HEALTH (ESTIMATED)
    # ==================================================
    st.markdown("---")
    st.subheader("🌿 Crop Health (Green Color of Crop)")
    st.caption(
        "Crop health is estimated dynamically from recent weather "
        "and water stress conditions."
    )

    ndvi_value = estimate_ndvi_from_idi(df)

    # NDVI interpretation
    if ndvi_value >= 0.6:
        st.success("🌱 Crop is healthy")
        ndvi_status = "Healthy vegetation"
    elif ndvi_value >= 0.45:
        st.warning("🌿 Crop is under moderate stress")
        ndvi_status = "Moderate stress"
    else:
        st.error("🚨 Crop is under severe stress")
        ndvi_status = "Severe stress"

    st.metric("🌿 Estimated NDVI", f"{ndvi_value:.2f}")

    st.caption(
        f"NDVI status: {ndvi_status}. "
        "Higher values indicate greener and healthier crops."
    )

    # ==================================================
    # IDI GRAPH
    # ==================================================
    st.markdown("---")
    st.subheader("📈 Water Stress Trend")

    fig_idi = px.line(df, x="date", y="idi_7day")
    fig_idi.add_hline(y=15, line_dash="dot", line_color="orange")
    fig_idi.add_hline(y=30, line_dash="dash", line_color="red")
    st.plotly_chart(fig_idi, use_container_width=True)

    # ==================================================
    # LEGEND
    # ==================================================
    st.markdown("""
### 🧭 Color Meaning
🟢 Green → Water is enough  
🟡 Yellow → Be careful  
🔴 Red → Irrigate now  
""")

    st.markdown(
        "<h3 style='text-align:center;'>👩‍🌾 Advice based on satellite weather data</h3>",
        unsafe_allow_html=True
    )
