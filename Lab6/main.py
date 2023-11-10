import arviz as az
import matplotlib.pyplot as plt
import pymc as pm

Y = [0, 5, 10]
THETA = [0.2, 0.5]

fig, axs = plt.subplots(3, 2, figsize=(14, 8), constrained_layout=True)

with pm.Model() as model:
    n = pm.Poisson('n', mu=10)

    for y in Y:
        for theta in THETA:
            dstr = pm.Binomial(f'dstr_posteriori_{y}_{theta}', n=n, p=theta, observed=y)
            posterior = pm.sample(2000, tune=1000, cores=1)
            az.plot_posterior(posterior, ref_val=0, ax=axs[Y.index(y), THETA.index(theta)])

plt.show()
