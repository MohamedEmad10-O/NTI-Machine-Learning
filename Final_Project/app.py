import json
from datetime import date

import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE = Path(__file__).parent

st.set_page_config(page_title="India Air Quality – RSPM Predictor", page_icon="🌫️", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(BASE / "models" / "rspm_model.pkl")          # Pipeline = preprocessing + XGBoost


@st.cache_data
def load_meta():
    with open(BASE / "app_metadata.json", encoding="utf-8") as f:
        return json.load(f)


model, meta = load_model(), load_meta()


def aqi_category(v):
    # تصنيف PM10 حسب الـ National AQI الهندي (24 ساعة)
    for limit, label, color in [(50, "Good", "#2e7d32"), (100, "Satisfactory", "#7cb342"),
                                (250, "Moderate", "#f9a825"), (350, "Poor", "#ef6c00"),
                                (430, "Very Poor", "#c62828")]:
        if v <= limit:
            return label, color
    return "Severe", "#6a1b9a"


st.title("🌫️ RSPM (PM10) Predictor")
st.caption("Predicts Respirable Suspended Particulate Matter (µg/m³) for Indian monitoring stations "
           "using a tuned XGBoost pipeline trained on CPCB data (1987–2015).")

with st.sidebar:
    st.header("Model performance (test set)")
    m = meta["metrics"]
    st.metric("R²", f"{m['R2']:.3f}")
    st.metric("MAE", f"{m['MAE']:.1f} µg/m³")
    st.metric("RMSE", f"{m['RMSE']:.1f} µg/m³")

st.subheader("Location")
c1, c2 = st.columns(2)
state = c1.selectbox("State", sorted(meta["states"]))
location = c2.selectbox("City / Location", sorted(meta["states"][state]))
info = meta["states"][state][location]

types = meta["types"]
area = st.selectbox("Area type", types, index=types.index(info["type"]) if info["type"] in types else 0)

st.subheader("Measurements")
c3, c4, c5 = st.columns(3)
so2 = c3.number_input("SO₂ (µg/m³)", min_value=0.0, max_value=1000.0, value=round(meta["so2_median"], 1))
no2 = c4.number_input("NO₂ (µg/m³)", min_value=0.0, max_value=1000.0, value=round(meta["no2_median"], 1))
d = c5.date_input("Date", value=date(2015, 1, 15), min_value=date(1987, 1, 1), max_value=date(2030, 12, 31))

with st.expander("Station details (filled automatically from the selected city)"):
    st.write(f"**Agency:** {info['agency'] or '—'}")
    st.write(f"**Monitoring station:** {info['station'] or '—'}")

if st.button("Predict RSPM", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "state": state, "location": location, "agency": info["agency"], "type": area,
        "so2": so2, "no2": no2, "location_monitoring_station": info["station"],
        "year": d.year, "month": d.month,
    }])
    pred = max(float(model.predict(row)[0]), 0.0)
    label, color = aqi_category(pred)
    st.markdown(
        f"<div style='padding:1.2rem;border-radius:12px;background:{color}22;border-left:6px solid {color}'>"
        f"<div style='font-size:0.95rem;opacity:.8'>Predicted RSPM</div>"
        f"<div style='font-size:2.4rem;font-weight:700'>{pred:.1f} µg/m³</div>"
        f"<div style='font-size:1.1rem;color:{color};font-weight:600'>{label}</div></div>",
        unsafe_allow_html=True)
    if d.year > 2015:
        st.info("The model was trained on data up to 2015 — predictions for later years are extrapolations.")
