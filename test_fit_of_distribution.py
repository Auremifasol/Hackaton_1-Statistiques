# Code here
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sc
from scipy.optimize import minimize


df = pd.read_csv('mortality_climate.csv')
# Garder le groupe d'age 45-64 et la saison hivernale.
df_filtered = df[(df['CD_AGEGROUP'] == '45-64') & (df['season'] == 'winter')].copy()

plt.figure(figsize=(10, 6))
plt.hist(
    df_filtered["MS_NUM_DEATH"],
    bins=np.arange( # regrouper les stats
        int(df_filtered["MS_NUM_DEATH"].min()),
        int(df_filtered["MS_NUM_DEATH"].max()) + 2, # +2 car le range ne vas pas jusqu'au dernier donc +1 et encore +1 pour avoir l'interval entier du dernier
    )-0.5,
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

x = df_filtered["MS_NUM_DEATH"].dropna().to_numpy()
n = len(x)
lam_hat = x.mean()

poisson_log_lik = sc.poisson.logpmf(x, lam_hat).sum()


#plt.show()

"""
Tout d'abord, on émet comme hypothèse que les variables aléatoires $x_{1}, ... x_{n}$ sont indépendantes, et on approxime par une loi de Poisson : $x_{i} ~ Poisson(\lambda)$
Pour une distribution de Poisson, la fonction de probabilité est grâce au formulaire : 
$$P(X=x) = \frac{e^{-\lambda}.\lambda^{x}}{x!}$$

On fait une approximation grâce à la méthode *maximum likelihood estimation (MLE)* : 
$$L(\lambda) = \prod_{i=1}^{n} \frac{e^{-\lambda}\,\lambda^{x_i}}{x_i!} = \frac{e^{-n\lambda}\,\lambda^{\sum_{i=1}^{n} x_i}}{\prod_{i=1}^{n} x_i!}
$$
On prend ensuite le logarithme :
$$ l(\lambda) = log(L(\lambda)) = -n\lambda + (\sum_{i=1}^{n} x_i) log(\lambda) - \sum_{i=1}^{n} log(x_{i}!)$$
On peut voir que l'expression est une forme analytique, on peut donc dériver pour trouver un maximum :
$$\frac{dl}{d\lambda} = -n + \frac{\sum{x_i = 1}^{n}}{\lambda} = 0$$
On trouve donc l'estimation suivante : 
$$ \hat{\lambda} = \bar{x}$$
"""

# -------------- fit Poisson (fin)---------------------------
# -------------- fit Negative Binomial MS_NUM_DEATH --------------------------------
"""observations = df_filtered["MS_NUM_DEATH"].to_numpy(dtype=int)
mean_deaths = observations.mean()

print("test sum logpmf : ", np.sum(sc.nbinom.logpmf(x)))
ajustement = sc.fit(
    sc.nbinom,
    x,
    bounds={"n": (0.01, 1000), "p": (0.001, 0.999)},
)
r_hat = ajustement.params.n
p_hat = ajustement.params.p"""

def neg_log_lik(params):
    r, p = params
    if r <= 0 or p <= 0 or p >= 1:
        return 1e12
    return -np.sum(sc.nbinom.logpmf(x, r, p))

# Initialisation par la méthode des moments
mean_x = lam_hat
var_x = np.var(x, ddof=1)

p0 = mean_x / var_x
r0 = (mean_x ** 2) / (var_x - mean_x)

res = minimize(neg_log_lik, [r0, p0], bounds=[(1e-5, None), (1e-5, 1 - 1e-5)])
r_hat, p_hat = res.x
nb_log_lik = -res.fun

# -------------- fit Negative Binomial (fin)---------------------------
# ------------------------------------ plot ------------------------------------

k = np.arange(x.min(), x.max() +1)
plt.figure(figsize=(10,6))
plt.xlabel("Nombre de décès par jour")
plt.ylabel("Densité")
plt.title("Ajustement de Poisson et binomiale négative (45-64 ans, hiver)")
# histograms
plt.hist(x, bins=np.arange(x.min(), x.max() +2) -0.5, density = True, alpha = 0.5, edgecolor = "black", label = "Données")
# fitted probability mass function Poisson
plt.plot(
    k,
    sc.poisson.pmf(k, lam_hat),
    "o-",
    color="tab:blue",
    label=f"Poisson (λ={lam_hat:.2f})",
)

# fitted probability mass function Negative Binomial
plt.plot(
    k,
    sc.nbinom.pmf(k, r_hat, p_hat),
    "s-",
    color="tab:orange",
    label=f"Binomiale négative (r={r_hat:.2f}, p={p_hat:.3f})",
)
plt.legend()
# ------------------------------------ plot ------------------------------------
# -------------------- compare log likelihood ----------------------------------
# and select the most apropriate

plt.show()
