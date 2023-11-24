import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import random 

# Problema 1

# Punctul a

# o functie de aruncare a monedei in functie de probabilitate (masluita sau nemasluita)
def throw_moneda(prob):
    return random.random() < prob

def simulation():
    first_player = random.choice([0, 1]) # decidem cine incepe jocul
    n = throw_moneda(1/3) # numarul de steme obtinute de primul jucator
    m = sum([throw_moneda(1/2) for _ in range(n + 1)]) # numarul de steme obtinute de al doilea jucator 
    if n >= m :
      return 0
    else:
      return 1

n = 20000
P0_wins = 0
P1_wins = 0

# simulam jocul de 20000 de ori
for _ in range(n):
    winner = simulation()
    if winner == 0:
        P0_wins += 1
    else:
        P1_wins += 1

print("Castiguri P0:", P0_wins)
print("Castiguri P1:", P1_wins)

# calculam si afisam procentele
P0_procentage = P0_wins / n* 100
P1_procentage = P1_wins / n *100

print(f"Procent castig P0: {P0_procentage}%")
print(f"Procent castig P1: {P1_procentage}%")

# Problema 2

# Punctul a
mu_a = 0
sigma_a = 1
waiting_time = np.random.normal(mu_a, sigma_a, 200) # general 200 de timpi medii de asteptare

# Punctul b
with pm.Model() as model:

    # din enunt: timpul mediu de asteptare este modelat  de o distributie normala de parametri mu si sigma
    mu = pm.Normal('mu', mu=0, sigma=1)
    sigma = pm.Normal('sigma', mu=0, sigma=1)
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=waiting_time)

    trace = pm.sample(2000, tune=2000, return_inferencedata=True)

az.plot_posterior(trace)
plt.show()
