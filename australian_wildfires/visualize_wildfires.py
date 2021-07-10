"""
File name: visualize_wildfires.py

Creation Date: Sat 10 Jul 2021

Description:
    Dataset: https://www.kaggle.com/carlosparadis/fires-from-space-australia-and-new-zeland

"""

# Python Libraries
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import gmaps
from ipywidgets.embed import embed_minimal_html
import webbrowser
import pathlib
from sklearn import preprocessing


# Local Application Modules
# -----------------------------------------------------------------------------


df = pd.read_csv("data/fire_archive_M6_96619.csv")
print(df.head())

lon = np.arange(df["longitude"].min(), df["longitude"].max())
lat = np.arange(df["latitude"].min(), df["latitude"].max())

scaler = preprocessing.MinMaxScaler()
df["brightness_norm"] = scaler.fit_transform(df["brightness"].values.reshape((-1, 1)))
print(df.head())

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


gmaps.configure(api_key=api_key)
fig = gmaps.figure()

fig.add_layer(gmaps.heatmap_layer(
    df[["latitude", "longitude"]], 
    weights=df["brightness_norm"]
    )
)
html_path = "visualization.html"
embed_minimal_html(html_path, views = [fig])


webbrowser.open('file://'+str(pathlib.Path().resolve())+"/"+html_path)

print(fig)
