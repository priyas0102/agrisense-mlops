def get_ndvi_points():
    import pandas as pd
    return pd.read_csv("ndvi_data.csv").to_dict("records")