# Code here
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mortality_climate.csv')
# Garder le groupe d'age 45-64 et la saison hivernale.
df_filtered = df[(df['CD_AGEGROUP'] == '45-64') & (df['season'] == 'winter')].copy()

plt.figure(figsize=(10, 6))
plt.hist(
    df_filtered["MS_NUM_DEATH"],
    bins=range( # regrouper les stats
        int(df_filtered["MS_NUM_DEATH"].min()),
        int(df_filtered["MS_NUM_DEATH"].max()) + 2, # +2 car le range ne vas pas jusqu'au dernier donc +1 et encore +1 pour avoir l'interval entier du dernier
    ),
    edgecolor="black", # sinon ilisible 
)
plt.xlabel("Nombre de décès par jour")
plt.ylabel("Nombre de jours")
plt.title("Décès quotidiens en hiver (45-64 ans)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# plt.figure(figsize=(10, 6))

# -------------- fit Poisson MS_NUM_DEATH --------------------------------
# avec max likelihood

# -------------- fit Poisson (fin)---------------------------
# -------------- fit Negative Binomial MS_NUM_DEATH --------------------------------

# -------------- fit Negative Binomial (fin)---------------------------
# ------------------------------------ plot ------------------------------------
# histograms
# fitted probability mass function Poisson
# fitted probability mass function Negative Binomial
# ------------------------------------ plot ------------------------------------

# -------------------- compare log likelihood ----------------------------------
# and select the most apropriate

plt.show()
