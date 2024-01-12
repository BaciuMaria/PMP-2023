import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Exercitiul 1
def posterior_grid(grid_points=50, heads=6, tails=9, prior_type="uniform"):
    grid = np.linspace(0, 1, grid_points)

    if prior_type == "uniform":
        prior = np.repeat(1 / grid_points, grid_points)  # uniform prior
    elif prior_type == "type1":
        prior = (grid <= 0.5).astype(int)
    elif prior_type == "type2":
        prior = abs(grid - 0.5)
    else:
        raise ValueError("Invalid prior_type")

    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


data = np.repeat([0, 1], (10, 3))
points = 20
h = data.sum()
t = len(data) - h

grid_uniform, posterior_uniform = posterior_grid(points, h, t, prior_type="uniform")
grid_binary, posterior_binary = posterior_grid(points, h, t, prior_type="type1")
grid_absolute, posterior_absolute = posterior_grid(points, h, t, prior_type="type2")

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(grid_uniform, posterior_uniform, 'o-')
plt.title(f'Uniform')
plt.yticks([])
plt.xlabel('θ')

plt.subplot(132)
plt.plot(grid_binary, posterior_binary, 'o-')
plt.title(f'Type 1')
plt.yticks([])
plt.xlabel('θ')

plt.subplot(133)
plt.plot(grid_absolute, posterior_absolute, 'o-')
plt.title(f'Type 2')
plt.yticks([])
plt.xlabel('θ')

plt.show()

# Exercitiul 2

def estimate_pi(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x ** 2 + y ** 2) <= 1
    pi = inside.sum() * 4 / N

    error = abs((pi - np.pi) / pi) * 100

    return error


num_runs = 1000
N_values = [100, 1000, 10000]

# stocam erorile intr-o matrice
errors_matrix = np.zeros((num_runs, len(N_values)))

for i in range(num_runs):
    for j, N in enumerate(N_values):
        errors_matrix[i, j] = estimate_pi(N)

mean_errors = np.mean(errors_matrix, axis=0)
std_errors = np.std(errors_matrix, axis=0)

plt.errorbar(N_values, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
plt.xlabel('Numărul de puncte (N)')
plt.ylabel('Eroare relativă (%)')
plt.title('Estimarea erorii relative în funcție de numărul de puncte')
plt.show()

# Exercitiul 3


def beta_binomial_posterior(x, a_prior, b_prior, y, N):
    prior = stats.beta.pdf(x, a_prior, b_prior)
    likelihood = stats.binom.pmf(y, N, x)
    posterior = prior * likelihood
    posterior /= posterior.sum()
    return posterior


def metropolis_beta_binomial(draws=10000, a_prior=1, b_prior=1):
    trace = np.zeros(draws)
    old_x = 0.5
    old_prob = beta_binomial_posterior(old_x, a_prior, b_prior, y=0, N=1)
    delta = np.random.normal(0, 0.5, draws)

    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = beta_binomial_posterior(new_x, a_prior, b_prior, y=0, N=1)
        acceptance = new_prob / old_prob

        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x

    return trace


a_prior, b_prior = 2, 5
prior_beta = stats.beta(a_prior, b_prior)
trace_metropolis = metropolis_beta_binomial(draws=10000, a_prior=a_prior, b_prior=b_prior)

plt.figure(figsize=(10, 6))

x_prior = np.linspace(0, 1, 100)
y_prior = prior_beta.pdf(x_prior)
plt.plot(x_prior, y_prior, 'C1-', lw=3, label='True distribution (Prior)')

plt.hist(trace_metropolis[trace_metropolis > 0], bins=25, density=True, alpha=0.7,
         label='Estimated distribution (Metropolis)')
plt.xlabel('θ')
plt.ylabel('pdf(θ)')
plt.yticks([])
plt.legend()

plt.xlim(0, 1)
plt.title('Distribuție a posteriori estimată cu Metropolis pentru modelul beta-binomial')
plt.show()