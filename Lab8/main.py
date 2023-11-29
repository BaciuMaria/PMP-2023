import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/file/d/1XrkXiIY5ZE3OX77BhT4w7MKdwx67pVfw/view'
file_id = link.split('/')[-2]
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile('Prices.csv')
data = pd.read_csv('Prices.csv')


price = data['Price']
speed= data['Speed']
harddrive = np.log(data['HardDrive'])

# Exercitiul 1

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + beta1 * speed + beta2 * harddrive

    y = pm.Normal('y', mu=mu, sigma=sigma, observed=price)

    trace = pm.sample(5000, tune=1000)

print(pm.summary(trace).round(2))
pm.plot_posterior(trace, var_names=['alpha', 'beta1', 'beta2', 'sigma'])

# Exercitiul 2

az.plot_posterior(trace, var_names=['beta1', 'beta2'], hdi_prob=0.95)
plt.show()