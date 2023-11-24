import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az

# Punctul a
mu_a = 0
sigma_a = 1
waiting_time = np.random.normal(mu_a, sigma_a, 200)

# Punctul b
with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sigma=1)
    sigma = pm.Normal('sigma', mu=0, sigma=1)

    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=waiting_time)

    trace = pm.sample(2000, tune=2000, return_inferencedata=True)

az.plot_posterior(trace)
plt.show()
