import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AgriSense AI",
    layout="wide"
)

# ---------------------------
# HEADER
# ---------------------------
st.title("🌱 AgriSense AI - Crop Health Monitoring System")
st.caption("Satellite-powered NDVI analysis with AI-based crop health prediction")

st.divider()

# ---------------------------
# SIDEBAR CONTROLS
# ---------------------------
st.sidebar.header("⚙️ Controls")

ndvi = st.sidebar.slider("Select NDVI Value", 0.0, 1.0, 0.5)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("crop_model.pkl")

def predict(ndvi):
    features = np.array([[ndvi, ndvi**2, np.log(ndvi + 1e-5)]])
    return model.predict(features)[0]

# ---------------------------
# METRICS DASHBOARD
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("NDVI Value", f"{ndvi:.2f}")

with col2:
    label = predict(ndvi)
    st.metric("Crop Status", label)

with col3:
    st.metric("System", "Active 🟢")

st.divider()

# ---------------------------
# PREDICTION BUTTON
# ---------------------------
if st.button("🌾 Analyze Crop Health"):
    with st.spinner("Running AI model..."):
        result = predict(ndvi)
        st.success(f"Prediction Result: {result}")

# ---------------------------
# MAP SECTION
# ---------------------------
st.subheader("🛰️ Satellite Crop Health Map")

m = folium.Map(location=[21.2, 81.6], zoom_start=11, tiles="OpenStreetMap")

# dummy marker example (replace with df later)
folium.CircleMarker(
    location=[21.2, 81.6],
    radius=10,
    popup=f"NDVI: {ndvi}",
    color="green",
    fill=True
).add_to(m)

st_folium(m, width=900, height=500)