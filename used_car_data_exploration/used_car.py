"""
File name: used_car.py

Creation Date: Sun 11 Jul 2021

Description:

"""

# Python Libraries
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


# Local Application Modules
# -----------------------------------------------------------------------------


df = pd.read_csv("data/vehicles.csv", nrows=10000)

# Prints number of unique values per variable, axis=1 would
# print number of unique values per sample

#print(df.nunique(axis=0))


# Describes distribution of each variable in dataframe: mean, max, min etc
print(df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))))

print(df.condition.unique())



## Plotting price distribution per condition variable
df.boxplot(column="price", by="condition")
for i, (k, d) in enumerate(df.groupby("condition")):
    y = d["price"]
    print(y)
    print(k)
    x = np.random.normal(i+1, 0.04, len(y))

    plt.plot(x, y, mfc = ["orange","blue","yellow", "red", "black", "purple"][i], mec='k', ms=7, marker="o", linestyle="None")
plt.tight_layout()
plt.savefig("output/price_per_condition.png")
plt.show()
