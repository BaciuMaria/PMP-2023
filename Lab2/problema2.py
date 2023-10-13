import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

alpha=[4,4,5,5]
beta=[1/3,1/2,1/2,1/3]


probabilitate_server=[0.25,0.25,0.30,0.20]
latenta_lambda=4
valori=[]


for _ in range(10000):
   # in functie de probabilitate alegem un server si calculam timpul total de servire a clientului
  index= np.random.choice(4,p=probabilitate_server)
  t_procesare=stats.gamma(alpha[index],scale=1/beta[index]).rvs()
  latenta=stats.expon(scale=1/latenta_lambda).rvs()
  t_total=t_procesare + latenta
  valori.append(t_total)

# probabilitatea ca timpul necesar sa fie mai mare decat 3
probabilitate=np.mean(np.array(valori)>3)
print(f"Probabilitatea ca X > 3 milisecunde:{probabilitate}")

# Grafic

plt.hist(valori, bins=50, density=True, alpha=0.6, color='b')
plt.xlabel('Timp de servire')
plt.ylabel('Densitate')
plt.title('Densitatea Distribuției lui X')
plt.show()