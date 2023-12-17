import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from sklearn.preprocessing import StandardScaler

az.style.use('arviz-darkgrid')

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/file/d/1umg5MZ1W0y7ZvF9-MKuNczXN1rCWhjBI/view'
file_id = link.split('/')[-2]
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile('dummy.csv')
dummy_data = pd.read_csv('dummy.csv', header=None, delimiter=' ')

print(dummy_data)

# Exercitiul 1
# Punctul a

x_1 = dummy_data.iloc[:, 0].values
y_1 = dummy_data.iloc[:, 1].values
order = 5
x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

beta_array = np.array([10, 0.1, 0.1, 0.1, 0.1])

with pm.Model() as model_p:
    alfa = pm.Normal('alfa', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=(order,))
    e = pm.HalfNormal('e', 5)
    mu = alfa + pm.math.dot(beta, x_1s)

    y_obs_array = pm.Normal('y_pred', mu=mu, sigma=e, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
alpha_p_post = idata_p.posterior['alfa'].mean(axis=(0, 1)).item()
beta_p_post = idata_p.posterior['beta'].mean(axis=(0, 1))
y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)
plt.plot(x_1s[0], y_p_post, 'C2', label=f'model order {order}, sigma=10')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Punctul b

with pm.Model() as model_l:
    alfa = pm.Normal('alfa', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=(order,))
    e = pm.HalfNormal('e', 5)
    mu = alfa + pm.math.dot(beta, x_1s)

    y_obs_array = pm.Normal('y_pred', mu=mu, sigma=e, observed=y_1s)
    idata_l = pm.sample(2000, return_inferencedata=True)

with pm.Model() as model_m:
    alfa = pm.Normal('alfa', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=beta_array, shape=(order,))
    e = pm.HalfNormal('e', 5)
    mu = alfa + pm.math.dot(beta, x_1s)

    y_obs_array = pm.Normal('y_pred', mu=mu, sigma=e, observed=y_1s)
    idata_m = pm.sample(2000, return_inferencedata=True)

x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

alpha_100 = idata_l.posterior['alfa'].mean(axis=(0, 1)).item()
beta_100 = idata_l.posterior['beta'].mean(axis=(0, 1))
y_p_100 = alpha_100 + np.dot(beta_100, x_1s)

alpha_array = idata_m.posterior['alfa'].mean(axis=(0, 1)).item()
beta_array = idata_m.posterior['beta'].mean(axis=(0, 1))
y_p_array = alpha_array + np.dot(beta_array, x_1s)

plt.plot(x_1s[0], y_p_100, 'C3', label=f'model order {order}, sigma=100')
plt.plot(x_1s[0], y_p_array, 'C4', label=f'model order {order}, sigma=array([10, 0.1, 0.1, 0.1, 0.1])')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Exercitiul 2

x_new = np.linspace(x_1.min(), x_1.max(), 500)
y_new = np.linspace(y_1.min(), y_1.max(), 500)

x_1 = np.concatenate([x_1, x_new])
y_1 = np.concatenate([y_1, y_new])

print("Generated x_1 values:", x_1)
print("Generated y_1 values:", y_1)

order = 5
x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

# print("Missing values in y_1s:", np.isnan(y_1s).sum())

y_1s_imputed = np.where(np.isnan(y_1s), np.nanmean(y_1s), y_1s)

with pm.Model() as model_l_new:
    alfa = pm.Normal('alfa', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=(order,))
    e = pm.HalfNormal('e', 5)
    mu = alfa + pm.math.dot(beta, x_1s)
    y_obs = pm.Normal('y_pred', mu=mu, sigma=e, observed=y_1s_imputed)
    idata_l_new = pm.sample(2000, return_inferencedata=True)

with pm.Model() as model_m_new:
    alfa = pm.Normal('alfa', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=beta_array, shape=(order,))
    e = pm.HalfNormal('e', 5)
    mu = alfa + pm.math.dot(beta, x_1s)
    y_obs = pm.Normal('y_pred', mu=mu, sigma=e, observed=y_1s_imputed)
    idata_m_new = pm.sample(2000, return_inferencedata=True)

x_new_plot = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
alpha_100_new = idata_l_new.posterior['alfa'].mean(axis=(0, 1)).item()
beta_100_new = idata_l_new.posterior['beta'].mean(axis=(0, 1))
y_100_new = alpha_100_new + np.dot(beta_100_new, x_1s)

alpha_array_new = idata_m_new.posterior['alfa'].mean(axis=(0, 1)).item()
beta_array_new = idata_m_new.posterior['beta'].mean(axis=(0, 1))
y_array_new = alpha_array_new + np.dot(beta_array_new, x_1s)

plt.plot(x_1s[0], y_100_new, 'C3', label=f'model order {order}, sigma=100')
plt.plot(x_1s[0], y_array_new, 'C4', label=f'model order {order}, sigma=array([10, 0.1, 0.1, 0.1, 0.1])')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
