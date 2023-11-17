import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
link = 'https://drive.google.com/file/d/1QFTznskcFmqNUXtx5fuMlisH3rSKG_Zj/view'
id = link.split('/')[-2]
print(id)

downloaded = drive.CreateFile({'id':id}) 
downloaded.GetContentFile('auto-mpg.csv')  
data = pd.read_csv('auto-mpg.csv')

print(data.head())

to_drop = ['cylinders', 'displacement', 'weight', 'acceleration', 'model year', 'origin', 'car name']
data_clean = data.drop(to_drop, axis=1)
x = data_clean['horsepower'].values
y = data_clean['mpg'].values

print(data_clean.head())

plt.scatter(data_clean['horsepower'], data_clean['mpg'])
plt.title('Relația dintre CP și mpg')
plt.xlabel('Cai putere (CP)')
plt.ylabel('Mile pe galon (mpg)')
plt.show()

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    epsilon = pm.HalfCauchy('epsilon', 5)

    mu= alpha + beta * x
    mpg_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y)

