import numpy as np

# Problema 1
def ex1(alpha):
    lambda_val = 20
    timp = 2
    dev_strd = 0.5
    clienti = np.random.poisson(lambda_val, size=1)

    timp_comanda = np.random.normal(timp, dev_strd, size=clienti)
    timp_gatit = np.random.exponential(alpha, size=clienti)
    timp_total = timp_comanda + timp_gatit
    comenzi = len(timp_total)

    timp_mediu = np.mean(timp_total)

    return clienti,comenzi,timp_mediu

alpha = 5
values = ex1(alpha)
print(f"Numărul de clienți: {values[0]}")
print(f"Numărul total de comenzile procesate într-o oră: {values[1]}")

# Problema 2

def ex2():
    lambda_val = 20
    timp = 2
    dev_strd= 0.5
    clienti = np.random.poisson(lambda_val)
    alphas = np.linspace(0.01,5.0,1000)
    probabilitati = []

    for alpha in alphas:
        max_timp_maxim = 0
        for _ in range(clienti):
            timp_comanda = np.random.normal(timp, dev_strd)
            timp_gatit = np.random.exponential(alpha)
            timp_total = timp_comanda + timp_gatit
            max_timp_maxim = max(max_timp_maxim, timp_total)

        if max_timp_maxim < 15:
            return alpha

alpha_max = ex2()
print("Valoarea maximă a lui α pentru a servi toți clienții în mai puțin de 15 minute:", alpha_max)

# Problema 3

alpha_max = ex2()
timp_mediu_asteptare = ex1(alpha_max)
print("Timpul mediu de așteptare pentru a fi servit al unui client:", timp_mediu_asteptare[2])