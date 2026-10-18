import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


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

#
# Question 2 :
#

# Séparer les données : training set (premiers 75%) & test set (derniers 25%)
n_lignes = len(df_reg)
n_lignes_split = int(0.75 * n_lignes)

training_set = df_reg.iloc[:n_lignes_split].copy() #Lignes de données : 0 à n_lignes_split-1
test_set = df_reg.iloc[n_lignes_split:].copy() #Lignes de données : n_lignes_split à n_lignes-1

# Définir les variables pour la régression
X_training = sm.add_constant(training_set[["Y_lag"]]) #Ajout de la constante bêta0 pour la régression
y_training = training_set["Y"]

X_test = sm.add_constant(test_set[["Y_lag"]]) #Ajout de la constante bêta0 pour la régression
y_test = test_set["Y"]

# Régression linéaire
model = sm.OLS(y_training, X_training).fit()

print(model.summary())

# Prédiction grâce aux données du training_set
y_prediction = model.predict(X_test)

plt.figure(figsize=(14, 5))
plt.plot(test_set["Datetime"], y_test, label="Actual consumption", alpha=0.8)
plt.plot(test_set["Datetime"], y_prediction, label="Predicted consumption", alpha=0.8)
plt.xlabel("Datetime")
plt.ylabel("Electricity consumption")
plt.title("Actual vs Predicted Electricity Consumption (Test Set)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#
# Comments : @Titouan
#

#
# Question 3 :
#

# Variables explicatives dont les variables de climat
features = ["Y_lag", "T_lag", "T_lag_sq"]

# Définir les variables pour la régression
X_train = sm.add_constant(training_set[features])
y_train = training_set["Y"]

X_test = sm.add_constant(test_set[features])
y_test = test_set["Y"]

# Régression linéaire
model_climate = sm.OLS(y_train, X_train).fit()

print(model_climate.summary())

# Prédiction grâce aux données du training_set
y_pred_climate = model_climate.predict(X_test)

plt.figure(figsize=(14, 5))
plt.plot(test_set["Datetime"], y_test, label="Actual consumption", alpha=0.8)
plt.plot(test_set["Datetime"], y_prediction, label="Predicted consumption", alpha=0.8)
plt.xlabel("Datetime")
plt.ylabel("Electricity consumption")
plt.title("Actual vs Predicted Electricity Consumption (Test Set)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#
# Comments : @Titouan
#

#
# Question 4 :
#

# Modèle sans climat 
# Erreur entre la vraie consommation et celle prédite 
erreur_sans = y_test - y_prediction

# Erreur absolue moyenne
MAE_sans = np.mean(np.abs(erreur_sans))

# Racine de l'erreur quadratique moyenne
RMSE_sans = np.sqrt (np.mean(erreur_sans**2))

# Modèle avec climat 
erreur_avec = y_test - y_pred_climate
MAE_avec = np.mean(np.abs(erreur_avec))
RMSE_avec = np.sqrt (np.mean(erreur_avec**2))

print ("Sans climat : MAE =", MAE_sans, "RMSE =", RMSE_sans)
print ("Avec climat : MAE =", MAE_avec, "RMSE =", RMSE_avec)

"""Question : "Does the inclusion of climate variables improve the predictive performance of the model?"
Réponse : Les valeurs obtenues (arrondi) sont : 
Sans climat : MAE = 928 et RMSE = 1259
Avec climat : MAE = 925 et RMSE = 1256
Les deux erreurs diminuent lorsqu'on regarde avec l'ajout des variables climatiques,
donc oui ces variables améliorent la performance prédictive du modèle.
Mais toutefois, cette différence entre les 2 modèles est très faible et ne représente même pas un demi pourcent.
"""