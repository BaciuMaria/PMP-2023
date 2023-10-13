import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

m1= 4
m2=6
probabilitate_m1=0.4

n= 10000

x1=np.random.exponential(scale= 1/m1, size= n)
x2=np.random.exponential(scale=1/m2,size = n)

m_selectat=np.random.choice([1,2],size= n ,p=[probabilitate_m1, 1 - probabilitate_m1])

valori=np.where(m_selectat == 1, x1,x2)

media=np.mean(valori)
deviatia_standard=np.std(valori)

# Grafic
plt.hist(valori, bins=50, density=True, alpha=0.6, color='b')
x = np.linspace(0, max(valori), 100)
pdf=stats.expon.pdf(x,scale=1/probabilitate_m1*m1+(1-probabilitate_m1)*m2)
plt.plot(x,pdf,'r-',lw=2)
plt.xlabel('X (Timpul de servire)')
plt.ylabel('Densitate')
plt.title('Densitatea Distribuției lui X')
plt.show()