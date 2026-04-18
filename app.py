import streamlit as st
import requests
import pandas as pd
import numpy as np
st.title("🌱 AgriSense AI - Satellite Crop Health Monitoring")
st.caption("MLOps + Remote Sensing + Machine Learning Dashboard")
st.success("🚀 System deployed successfully on Streamlit Cloud")
st.sidebar.info("Built using Streamlit + ML + Satellite NDVI Data")
import folium
from streamlit_folium import st_folium
import joblib
import ee
from ee_data import get_ndvi_points
model = joblib.load("crop_model.pkl")


def predict_label(ndvi):
    features = np.array([[ndvi, ndvi**2, np.log(ndvi + 1e-5)]])
    return model.predict(features)[0]
# ---------------------------
# Load ML Model
# ---------------------------
model = joblib.load("crop_model.pkl")

def predict_label(ndvi):
    features = np.array([[ndvi, ndvi**2, np.log(ndvi + 1e-5)]])
    return model.predict(features)[0]

def get_color(label):
    if label == "Healthy":
        return "green"
    elif label == "Moderate":
        return "orange"
    else:
        return "red"
from ee_data import get_ndvi_points
df = pd.read_csv("ndvi_data.csv")
@st.cache_data
def load_data():
    data = get_ndvi_points()
    df = pd.DataFrame(get_ndvi_points())
    return pd.DataFrame(data)

df = load_data()
st.title("🌱 AgriSense")

ndvi = st.slider("NDVI Value", 0.0, 1.0, 0.5)

if st.button("Predict"):
    features = np.array([[ndvi, ndvi**2, np.log(ndvi + 1e-5)]])
    result = model.predict(features)[0]
    st.success(result)
# ---------------------------
# UI
# ---------------------------
st.title("🌱 AgriSense - Crop Health Monitoring")

st.write("App started ✔")
st.subheader("🌿 Test Crop Health Prediction")

ndvi = st.slider("Select NDVI Value", 0.0, 1.0, 0.5)

if st.button("Predict Crop Health"):
    result = predict_label(ndvi)
    st.success(f"Prediction: {result}")

# ---------------------------
# Dummy NDVI data (SAFE - NO EARTH ENGINE)
# ---------------------------
def load_data():
    return pd.DataFrame({
        "lat": [21.20, 21.21, 21.22, 21.23, 21.24],
        "lon": [81.60, 81.61, 81.62, 81.63, 81.64],
        "ndvi": [0.85, 0.60, 0.30, 0.75, 0.20]
    })

st.write("Loading data...")

df = load_data()

st.write("Data preview:")
st.write(df)

# ---------------------------
# Satellite Map
# ---------------------------
m = folium.Map(
    location=[21.2, 81.6],
    zoom_start=11,
    tiles=None
)

# Satellite layer
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
).add_to(m)

# Add markers
for _, row in df.iterrows():
    label = predict_label(row["ndvi"])

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        color=get_color(label),
        fill=True,
        fill_opacity=0.7,
        popup=f"NDVI: {row['ndvi']} | {label}"
    ).add_to(m)

# Layer control
folium.LayerControl().add_to(m)

# ---------------------------
# Show map
# ---------------------------
import branca.colormap as cm
from folium.plugins import HeatMap

# ---------------------------
# Create base map (satellite)
# ---------------------------
m = folium.Map(
    location=[21.2, 81.6],
    zoom_start=11,
    tiles=None
)

# Satellite layer
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
).add_to(m)

# OpenStreetMap layer
folium.TileLayer("OpenStreetMap").add_to(m)

# ---------------------------
# NDVI Heatmap Layer
# ---------------------------
heat_data = [[row["lat"], row["lon"], row["ndvi"]] for _, row in df.iterrows()]

HeatMap(
    heat_data,
    name="NDVI Heatmap",
    radius=18,
    blur=12,
    min_opacity=0.3
).add_to(m)

# ---------------------------
# Color scale legend
# ---------------------------
colormap = cm.LinearColormap(
    colors=["red", "orange", "green"],
    vmin=0,
    vmax=1,
    caption="NDVI Health Index"
)

colormap.add_to(m)

# ---------------------------
# ML markers (still there)
# ---------------------------
for _, row in df.iterrows():
    label = predict_label(row["ndvi"])

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        color=get_color(label),
        fill=True,
        fill_opacity=0.8,
        tooltip=f"""
        NDVI: {row['ndvi']:.2f}
        Status: {label}
        """
    ).add_to(m)

# ---------------------------
# Layer control
# ---------------------------
folium.LayerControl().add_to(m)

# ---------------------------
# Show map
# ---------------------------
st.subheader("🛰️ Advanced Crop Health Map")
st_folium(m, width=750, height=550)
st.subheader("🛰️ Crop Health Map")
st_folium(m, width=700, height=500)
m = folium.Map(location=[21.2, 81.6], zoom_start=11)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=6,
        fill=True,
        popup=f"NDVI: {row['ndvi']}"
    ).add_to(m)

st_folium(m, width=700, height=500)