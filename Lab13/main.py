import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

# Exercitiul 1

data_1= az.load_arviz_data("centered_eight")

data_2= az.load_arviz_data("non_centered_eight")

print(f"Numărul de lanțuri pentru modelul centrat: {data_1.posterior.chain.size}")
print(f"Mărimea totală a eșantionului pentru modelul centrat: {data_1.posterior.chain.size}")

print(f"Numărul de lanțuri pentru modelul necentrat: {data_2.posterior.draw.size}")
print(f"Mărimea totală a eșantionului pentru modelul necentrat: {data_2.posterior.draw.size}")

az.plot_posterior(data_1, var_names=["mu", "tau"], point_estimate="mean")
plt.suptitle("Centered Eight")
plt.show()

az.plot_posterior(data_2, var_names=["mu", "tau"], point_estimate="mean")
plt.suptitle("Non-Centered Eight")
plt.show()

# Exercitiul 2
summaries = pd.concat([az.summary(data_1, var_names=['mu']), az.summary(data_2, var_names=['mu'])])
summaries.index = ['centered', 'non_centered']
print(summaries)

summaries1 = pd.concat([az.summary(data_1, var_names=['tau']), az.summary(data_2, var_names=['tau'])])
summaries1.index = ['centered', 'non_centered']
print(summaries1)