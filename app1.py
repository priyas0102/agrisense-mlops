import joblib
import numpy as np

# load model
model = joblib.load("crop_model.pkl")


def predict_label(ndvi):
    features = np.array([[ndvi, ndvi ** 2, np.log(ndvi + 1e-5)]])
    return model.predict(features)[0]


def get_color(label):
    if label == "Healthy":
        return "green"
    elif label == "Moderate":
        return "orange"
    else:
        return "red"


# Map
m = folium.Map(location=[21.2, 81.6], zoom_start=11)

for _, row in df.iterrows():
    label = predict_label(row["ndvi"])

    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=5,
        color=get_color(label),
        fill=True,
        fill_opacity=0.7,
        popup=f"NDVI: {row['ndvi']:.2f} | {label}"
    ).add_to(m)