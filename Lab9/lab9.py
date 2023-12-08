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
link = 'https://drive.google.com/file/d/1nj2PykVs_VU92S9_tLi_xBaj61u2j46M/view'
file_id = link.split('/')[-2]
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile('Admission.csv')
data = pd.read_csv('Admission.csv')

scaler = StandardScaler()
data[['GRE', 'GPA']] = scaler.fit_transform(data[['GRE', 'GPA']])
gre = data['GRE'].values
gpa = data['GPA'].values
admission = data['Admission'].values

# Problema 1

with pm.Model() as model:
    beta0 = pm.Normal("beta0", mu=0, sigma = 10)
    beta1 = pm.Normal("beta1", mu=0, sigma = 10)
    beta2 = pm.Normal("beta2", mu=0, sigma = 10)
    μ = beta0 + beta1 * gre + beta2 * gpa
    θ = pm.Deterministic('θ', pm.math.sigmoid(μ))

    yl = pm.Bernoulli('yl', p=θ, observed=admission)

with model:
    trace = pm.sample(2000, tune=1000, chains=2)

pm.summary(trace).round(2)
plt.show()
print(pm.summary(trace))
az.plot_pair(trace, var_names=['beta0', 'beta1', 'beta2'])
plt.show()

# Problema 2

posterior = trace.posterior.stack(samples=("chain", "draw"))

x_1 = np.column_stack((gre, gpa))
idx = np.argsort(gre)

beta0_mean = posterior["beta0"].mean("samples").values
beta1_mean = posterior["beta1"].mean("samples").values
beta2_mean = posterior["beta2"].mean("samples").values

xx, yy = np.meshgrid(np.linspace(gre.min(), gre.max(), 100), np.linspace(gpa.min(), gpa.max(), 100))

decision_boundary = -(beta0_mean + beta1_mean * xx + beta2_mean * yy) / beta2_mean
hdi_94 = pm.hdi(decision_boundary, hdi_prob=0.94)

plt.scatter(gre, gpa, c=admission)
plt.fill_between(xx[0], hdi_94[:, 0], hdi_94[:, 1], color='red', alpha=0.2, label='Interval HDI 94%')
plt.plot(xx[0], decision_boundary, color='k', linewidth=1, label='Granița de Decizie')
plt.title("Granița de decizie și Intervalul HDI")
plt.xlabel("GRE")
plt.ylabel("GPA")
plt.show()