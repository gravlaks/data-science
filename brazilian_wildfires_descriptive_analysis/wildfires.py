"""
File name: wildfires.py

Creation Date: Sat 10 Jul 2021

Description:
    source: https://www.kaggle.com/andradaolteanu/unbiased-look-on-brazil-wildfires
    dataset: https://www.kaggle.com/gustavomodelli/forest-fires-in-brazil

"""

# Python Libraries
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# Local Application Modules
# -----------------------------------------------------------------------------


df = pd.read_csv("data/amazon.csv")

print(df.head())
number_stats = df["number"].describe().reset_index()
#number_stats.style.format({"number": "{:20,.0f}"}).hide_index().highlight_max(color="red")

def translate_months(month):
    months = {
            "Janeiro": "Jan", 
            "Fevereiro": "Feb",
            "Mar�o": "Mar",
            "Abril": "Apr", 
            "Maio": "May", 
            "Junho": "Jun", 
            "Julho": "Jul", 
            "Agosto": "Aug", 
            "Setembro": "Sep", 
            "Outubro": "Oct", 
            "Novembro": "Nov", 
            "Dezembro": "Dez" 
        }
    return months[month]
df["month"] = df["month"].apply(translate_months)
year_month_state = df.groupby(["year", "month", "state"]).sum().reset_index()
print(year_month_state.head())

years = df.groupby(["year"]).sum().reset_index()

print(years.head())
print(years.info())
#years.plot.line(x="year", y="number")
sns.lmplot(x="year", y="number", data=years, fit_reg=True)
plt.title("Wildfire distribution by year")
plt.tight_layout()
plt.savefig("output/wildfires_distribution_by_year.png")

plt.figure()
months = df.groupby(["month"]).sum().reset_index()
print(months)
sns.boxplot(x="month", y="number",
        order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Oct", "Nov", "Dec"],
        data=year_month_state, palette="coolwarm", saturation=1, width = 0.9, fliersize=4, linewidth=2)


plt.title("Wildfire distribution by month")
plt.tight_layout()
plt.savefig("output/wildfires_distribution_by_month.png")

#Top 10 states

top_10_states = df.groupby("state").sum().reset_index().sort_values(by="number", ascending=False).head(10)
top_10_states.plot.bar(x="state", y="number")
plt.title("Top 10 states by number of wildfires")
plt.tight_layout()
plt.savefig("output/top_10_states.png")


#Line plots for wildfires per year for each state
years_state = df.groupby(["year", "state"]).sum().reset_index()
years_state = years_state.where(
        (years_state["state"] == "Amazonas") |
        (years_state["state"] == "Mato Grosso") |
        (years_state["state"] == "Paraiba" )
        ).dropna()
print(years_state.info())
years_state = years_state.pivot(index="year", columns="state", values="number")
years_state.plot()
plt.title("Wildfires main states")
plt.tight_layout()
plt.savefig("output/wildfires_main_states.png")

plt.show()
