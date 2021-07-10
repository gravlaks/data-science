"""
File name: corona_virus.py

Creation Date: Sat 10 Jul 2021

Description: Visualize corona virus statistics with choropleth map
Source : https://towardsdatascience.com/visualizing-the-coronavirus-pandemic-with-choropleth-maps-7f30fccaecf5
Dataset: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset

"""

# Python Libraries
# -----------------------------------------------------------------------------

# Local Application Modules
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)


df = pd.read_csv("data/covid_19_data.csv")

df = df.rename(columns = {'Country/Region': 'Country'})
df = df.rename(columns = {'ObservationDate': 'Date'})

df_countrydate = df.groupby(["Date", "Country"]).sum().reset_index()


df_countrydate["Date"] = pd.to_datetime(df_countrydate["Date"], yearfirst = True)
df_countrydate.sort_values(by="Date", inplace=True)

df_countrydate["Date"] = df_countrydate.Date.dt.strftime("%Y-%d-%m")


#print(df_countrydate.where(df_countrydate["Country"] == "Spain").dropna().iloc[56:])
fig = px.choropleth(df_countrydate,
            locations = "Country",
            locationmode = 'country names', 
            color = "Confirmed",
            hover_name = 'Country', 
            animation_frame = "Date",
        )

fig.update_layout(
        title_text = 'Global spread of Covid 19', 
        title_x = 0.5, 
        geo = dict(
            showframe = False, 
            showcoastlines = False, 
            projection_type = 'equirectangular'
            )
        )
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 30
fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 5
fig.update_geos(projection_type="equirectangular", visible=True, resolution=110)


fig.show()
