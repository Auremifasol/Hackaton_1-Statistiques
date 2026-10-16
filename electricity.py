import pandas as pd
import matplotlib.pyplot as plt


"""
The goal is to predict electricity consumption at time $t$ using its past values and climate features. Let $Y_t$ denote electricity consumption at time $t$. We consider the following regression model:

$$Y_t = \beta_0 + \beta_1 Y_{t-7Days} + \beta_2 \bar{T}_{t-7Days} + \beta_3 \bar{T}_{t-7Days}^2 + \varepsilon_t,$$

where $Y_{t-7Days}$ denotes electricity consumption observed 7 days earlier, $\bar{T}_{t-7Days}$ is the average temperature observed 7 days earlier$^1$, and $\varepsilon_t \sim \mathcal{N}(0,\sigma^2)$ is an error term.

$^1$ The average temperature is measured at a daily frequency; therefore, $\bar{T}_{t-7Days}$ is constant for all time points $t$ within the same day.

**5.1)** Load the dataset *electricity_climate.csv* and construct the dataset required to perform the regression.

**Warning:** the data are observed at 15-minute intervals.

To build the regression dataset, each row must contain:
- electricity consumption at time $t$: $Y_t$,
- electricity consumption at time $t - 7$ days: $Y_{t-7Days}$,
- temperature at time $t - 7$ days: $\bar{T}_{t-7Days}$,
- the squared temperature at time $t - 7$ days: $\bar{T}_{t-7Days}^2$.
"""
#
# Question 1 :
#

df = pd.read_csv("electricity_climate.csv")

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime").drop_duplicates(subset="Datetime").reset_index(drop = True) #trier dans l'ordre chonologique même si c'est peut-être déjà fait

lag = 7 * 24 * 4 # = 7 (jours) * 24 (heures) * 4 (observations/heure) = 672 observations => prédictions se font avec observations faites 7 jours avant

print(df["Datetime"].diff().value_counts())

# Instancier les variables retardées de 7 jours:

df["Y_lag"] = df["Consumption"].shift(lag)
df["T_lag"] = df["temp_avg"].shift(lag)
df["T_lag_sq"] = df["temp_avg_sq"].shift(lag)

# Créer la dataset de la régression :

df_reg = df[["Datetime", "Consumption", "Y_lag", "T_lag", "T_lag_sq"]].dropna().rename(columns={"Consumption" : "Y"}).reset_index(drop = True)  

print(df_reg.head())
print(df_reg.shape)