import ee

# Initialize once
ee.Initialize(project="agrisense-mlops")

def get_ndvi_points():

    roi = ee.Geometry.Rectangle([81.5, 21.0, 81.8, 21.4])

    image = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate("2023-01-01", "2023-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
    )

    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    samples = ndvi.sample(
        region=roi,
        scale=10,
        numPixels=200,
        geometries=True
    )

    features = samples.getInfo()["features"]

    data = []

    for f in features:
        if "coordinates" not in f["geometry"]:
            continue

        lon, lat = f["geometry"]["coordinates"]
        val = f["properties"]["NDVI"]

        data.append({
            "lat": lat,
            "lon": lon,
            "ndvi": val
        })

    return data