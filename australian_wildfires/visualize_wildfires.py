"""
File name: visualize_wildfires.py

Creation Date: Sat 10 Jul 2021

Description:

"""

# Python Libraries
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Local Application Modules
# -----------------------------------------------------------------------------


df = pd.read_csv("data/fire_archive_M6_96619.csv")
print(df.head())

lon = np.arange(df["longitude"].min(), df["longitude"].max())
lat = np.arange(df["latitude"].min(), df["latitude"].max())



load_dotenv()

print(os.getenv("GOOGLE_API_KEY"))
