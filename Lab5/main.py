import pymc3 as pm
import csv

with open("./trafic.csv", 'r') as file:
  data = csv.reader(file)
  next(data)
  values = [int(row[1]) for row in data]

time = pm.DiscreteUniform('time', lower=4*60,upper=24*60)

with pm.Model()as our_first_model:
    alpha = 1.0
    lambda_1 = pm.Exponential('lambda_1', alpha)
    lambda_2 = pm.Exponential('lambda_2', alpha)
    lambda_3 = pm.Exponential('lambda_3', alpha)
    lambda_4 = pm.Exponential('lambda_4', alpha)
    lambda_5 = pm.Exponential('lambda_5', alpha)

    lambda_values = pm.switch(time < 7 * 60, lambda_1,pm.switch(time < 8 * 60, lambda_2,pm.switch(time < 16 * 60, lambda_3,pm.switch(time < 19 * 60, lambda_4, lambda_5))))

    trafic = pm.Poisson('trafic', mu=lambda_values, observed=values)

with our_first_model:
    trace=pm.sample(10000,tune=5000,random_seed=123,return_interference=True)

lambda_samples= trace['lambda']