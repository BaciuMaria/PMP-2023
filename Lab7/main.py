import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from sklearn.preprocessing import StandardScaler

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/file/d/1QFTznskcFmqNUXtx5fuMlisH3rSKG_Zj/view'
id = link.split('/')[-2]
print(id)

# Exercitiul 1

downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('auto-mpg.csv')
data = pd.read_csv('auto-mpg.csv')

print(data.head())

to_drop = ['cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'origin', 'car name']
data_clean = data.drop(to_drop, axis=1)
data_clean.replace('?', np.nan, inplace=True)
data_clean = data_clean.dropna()
data_clean = data_clean.astype(float)
data_clean = data_clean[data_clean['mpg'] > 0]

print(data_clean.head())

plt.scatter(data_clean['horsepower'], data_clean['mpg'])
plt.title('Relația dintre CP și mpg')
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile pe galon (mpg)')
plt.show()

# Exercitiul 2

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=1)
    epsilon = pm.HalfCauchy('epsilon', 5)

    mu = alpha + beta * data_clean['horsepower']
    mpg = pm.Normal('mpg', mu=mu, sigma=epsilon, observed=data_clean['mpg'])

    idata_g = pm.sample(2000, tune=2000, return_inferencedata=True)

# Exercitiul 3

plt.plot(data_clean['horsepower'], data_clean['mpg'], 'C0.')
posterior_g = idata_g.posterior.stack(samples={"chain", "draw"})
alpha_m = posterior_g['alpha'].mean().item()
beta_m = posterior_g['beta'].mean().item()
draws = range(0, posterior_g.samples.size, 10)
plt.plot(data_clean['horsepower'],
         posterior_g['alpha'][draws].values + posterior_g['beta'][draws].values * data_clean['horsepower'][:, None],
         c='gray', alpha=0.5)
plt.plot(data_clean['horsepower'], alpha_m + beta_m * data_clean['horsepower'], c='k',
         label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.legend()

# Exercitiul 4

pm.summary(idata_g)

az.plot_posterior(idata_g, hdi_prob=0.95, var_names=['alpha', 'beta'])
plt.scatter(data_clean['horsepower'], data_clean['mpg'], color='blue', alpha=0.5, label='Observations')
plt.title('Relația dintre Cai Putere și Consumul de Combustibil (mpg) cu Regiunea 95%HDI')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.legend()
plt.show()
