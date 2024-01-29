import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

# Exercitiul 1

# Punctul a

data= pd.read_csv("Titanic.csv")

# din enunt intelegem ca avem nevoie doar de datele din coloana Survived, Pclass si Age asa ca putem da drop la restul
to_drop = ['PassengerId','Name', 'Sex', 'SibSp','Parch','Ticket','Fare','Cabin','Embarked']
data_clean = data.drop(to_drop, axis=1)
#gestionarea datelor lipsa
data_clean.replace('NaN', np.nan, inplace=True)
data_clean = data_clean.dropna()
# transformarea variabilele float in int
data_clean = data_clean.astype(int)

print(data_clean.head())
pclass= data_clean["Pclass"].values
age = data_clean["Age"].values
survived = data_clean["Survived"].values

# Punctul b
# Definiti modelul in PyMC folosind doua variabile independente ( clasa Pclass si Age ) pentru a preciza variabila dependenta ( Survived ).

with pm.Model() as model_a:
    alfa = pm.Normal('alfa', mu=0, sigma=10)
    beta_1= pm.Normal('beta_1', mu=0, sigma=10)
    beta_2 = pm.Normal('beta_2', mu=0, sigma=10)
    mu = pm.Deterministic('μ',alfa + beta_1 * pclass + beta_2 * age)
    theta = pm.Deterministic("theta", pm.math.sigmoid(mu))
    bd = pm.Deterministic("bd", -alfa/beta_2 - beta_1/beta_2 * pclass)
    y_pred = pm.Bernoulli("y_pred", p=theta, observed=survived)

    idata_a = pm.sample(1250, return_inferencedata=True)

# Punctul d
# Construim un interval de 90% HDI pentru probabilitatea de a supravietui a unui pasager de 30 de ani de la clasa a 2-a.
posterior_g = idata_a.posterior.stack(samples={"chain", "draw"})
mu = posterior_g['alfa']+2*posterior_g['beta_1']+30*posterior_g['beta_2']
az.plot_posterior(mu.values,hdi_prob=0.9)

# Exercitul 2

# Punctul a

# numarul de iteratii
N = 10000
# variabilele aleatoare independente X si Y
x = np.random.geometric(0.3, size= N)
y = np.random.geometric(0.5, size= N)
# probabilitatea
p = np.sum(x > y**2)/N

plt.figure(figsize=(8, 8))
plt.plot(x[x > y**2], y[x > y**2], 'b.')
plt.plot(x[np.invert(x > y**2)], y[np.invert( x > y**2 )], 'r.')
plt.axis('square')
plt.xticks([])
plt.yticks([])
plt.legend(loc=1, frameon=True, framealpha=0.9)

print(f'Probabilitatea aproximată P(x > y^2): {p}')
