import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# Bonus 1
def ex1(alpha, num_samples):
    lambda_val = 20
    timp = 2
    dev_strd = 0.5
    timp_mediu_asteptare = []

    for _ in range(num_samples):
        clienti = np.random.poisson(lambda_val)
        timpi_comanda = np.random.normal(timp, dev_strd, size=clienti)
        timpi_gatit = np.random.exponential(alpha, size=clienti)
        timpi_total = timpi_comanda + timpi_gatit
        timp_mediu = np.mean(timpi_total)
        timp_mediu_asteptare.append(timp_mediu)

    return timp_mediu_asteptare

alpha = 3
num_samples = 100
timp_mediu_asteptare_samples = ex1(alpha, num_samples)

# Bonus 2

with pm.Model() as model:
    alpha = pm.Exponential('alpha', lam=1.0)
    obs = pm.Exponential("obs", lam=alpha, observed=timp_mediu_asteptare_samples)
    trace = pm.sample(2000,cores=2)

plt.figure(figsize=(8, 6))
plt.hist(trace["alpha"], bins=30, density=True, alpha=0.5, color='b')
plt.title("Distribuția estimată pentru α")
plt.xlabel("Valoarea lui α")
plt.ylabel("Densitate")
plt.show()