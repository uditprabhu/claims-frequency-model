"""
Interactive demo: enter a driver's details, get their predicted claim frequency.

Run (after `python -m src.train`):
    streamlit run app.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "models.joblib"

st.set_page_config(page_title="Claim Frequency Predictor", page_icon="🚗")
st.title("Motor insurance claim frequency predictor")
st.caption("Poisson GLM vs gradient boosting, trained on ~680k French motor "
           "policies (freMTPL2).")

if not MODEL_PATH.exists():
    st.error("No trained models found. Run `python -m src.train` first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
meta = bundle["meta"]
if meta["synthetic"]:
    st.warning("These models were trained on SYNTHETIC test data.")

cats, rng = meta["categories"], meta["ranges"]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Driver")
    driv_age = st.slider("Driver age", 18, 90, 35)
    bonus_malus = st.slider(
        "Bonus-malus score", 50, 150, 50,
        help="French no-claims score: 50 = best possible record, "
             "100 = new driver, higher = more past claims.")
    area = st.selectbox("Area (A = rural … F = urban)", cats["Area"])
    density = st.number_input("Population density (people/km²)",
                              min_value=1, max_value=30_000, value=500)
    region = st.selectbox("Region code", cats["Region"])
with col2:
    st.subheader("Vehicle")
    veh_age = st.slider("Vehicle age (years)", 0, 20, 5)
    veh_power = st.slider("Vehicle power band", 4, 15, 6)
    veh_brand = st.selectbox("Vehicle brand code", cats["VehBrand"])
    veh_gas = st.radio("Fuel", cats["VehGas"], horizontal=True)

row = pd.DataFrame([{
    "Area": area, "VehBrand": veh_brand, "VehGas": veh_gas, "Region": region,
    "VehPower": veh_power, "VehAge": veh_age, "DrivAge": driv_age,
    "BonusMalus": bonus_malus, "LogDensity": np.log(density),
}])

glm_pred = float(bundle["glm"].predict(row)[0])
gbm_pred = float(bundle["gbm"].predict(row)[0])
avg = meta["portfolio_frequency"]

st.divider()
st.subheader("Expected claims per year")
c1, c2, c3 = st.columns(3)
c1.metric("Poisson GLM", f"{glm_pred:.3f}",
          f"{glm_pred / avg:.2f}× average", delta_color="off")
c2.metric("Gradient boosting", f"{gbm_pred:.3f}",
          f"{gbm_pred / avg:.2f}× average", delta_color="off")
c3.metric("Portfolio average", f"{avg:.3f}")

st.caption(f"A frequency of {gbm_pred:.3f} means roughly one claim every "
           f"{1 / gbm_pred:.0f} years for a driver like this. A pricing team "
           "would multiply this by the expected cost per claim to get the "
           "risk premium.")
