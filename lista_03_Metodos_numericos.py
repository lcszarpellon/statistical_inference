# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 20:13:06 2025

@author: lucas
"""

### 01 - Resolução de integral

import numpy as np
from numpy.polynomial.laguerre import laggauss

lambda_ = 7867
n_points = 21

def f(u):
    t = u / 0.12
    return (lambda_**3 - t) / 0.12

# Posições e pesos para quadratura de Gauss-Laguerre
x, w = laggauss(n_points)

# Integral aproximada
integral = np.sum(w * f(x))
print(f"{integral:.3f}")

#%%
### 03 - Derivada numérica

def f(x):
    return 7*x**2 + 6*x + 2

def segunda_derivada_central(f, x, h):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

x = 1
h = 0.001

resultado = segunda_derivada_central(f, x, h)
print(f"{resultado:.3f}")
#%%
### 05 - Derivada numérica

import math

def f(x):
    return x**2 * math.exp(x) * math.tan(x)

x = 1
h = 0.001

result = segunda_derivada_central(f, x, h)
print(f"{result:.3f}")

#%%
### 07 - Resolução de integral

np.random.seed(123)

kappa = 2
n_points = 975

def integrand(x, kappa):
    return 1/(kappa + 1/kappa) * np.exp(-np.abs(x)/kappa)

samples = np.random.laplace(loc=0, scale=kappa, size=n_points)

values = integrand(samples, kappa)


integral_estimate = np.mean(values)

print(f"Estimativa da integral: {integral_estimate:.3f}")

#%%
### 08 - Resolução de integral

a = 0.1
b = 10.1
lam = 12
k = 5
n_points = 21

def integrand(y, lam, k):
    return k * lam * (y / lam)**(k - 1) * np.exp(- (y / lam)**k)


x, w = np.polynomial.legendre.leggauss(n_points)


t = 0.5 * (x + 1) * (b - a) + a


f_t = integrand(t, lam, k)


integral = 0.5 * (b - a) * np.dot(w, f_t)

print(f"{integral:.3f}")

#%%
### 09 - Integral usando quadratura Gaussiana

from numpy.polynomial.laguerre import laggauss

lam = 9
n_points = 29

def f(x):
    return lam * np.sqrt(x)


x, w = laggauss(n_points)

integral = np.sum(w * f(x))

print(f"{integral:.3f}")

#%%
### 10 - Raiz usando método de Newton


y = np.array([3.390, 7.379, 1.630, 4.778, 8.874, 5.531, 2.216, 1.582, 7.471, 5.296])
eps = 0.0001

def f(mu):
    return np.sum(np.exp(mu) * (mu - 1 - y) + np.exp(mu))

def df(mu):
    return np.sum(np.exp(mu) * (mu - y + mu))

def newton(mu0):
    mu = mu0
    while True:
        f_mu = f(mu)
        df_mu = df(mu)
        mu_new = mu - f_mu / df_mu
        if abs(mu_new - mu) < eps:
            break
        mu = mu_new
    return mu

raiz = newton(mu0=6)
print(f"{raiz:.3f}")
#%%
### 11 - Integral


np.random.seed(123)
kappa = 0.5
n = 1088

# Amostragem da distribuição exponencial com média kappa
x = np.random.exponential(scale=kappa, size=n)
# Função integrando: 1 / (kappa + kappa^-1) * exp(-x / kappa)
integrando = 1 / (kappa + 1 / kappa) * np.exp(-x / kappa)

# Estimativa da integral como média simples
estimativa = np.mean(integrando)

print(f"{estimativa:.3f}")

#%%
### 12 - Integral usando quadratura Gaussiana
from numpy.polynomial.hermite import hermgauss


lmbda = 0.055
n_points = 25


x, w = hermgauss(n_points)

integral = np.sum(w * lmbda * np.exp(2 * lmbda * x - lmbda**2))

round(float(integral), 3)

#%%
### 13 - Gradiente descendente


y = np.array([10.179, 10.073, 10.505, 10.022, 10.041, 10.557, 10.147, 10.408, 9.785, 9.860])
epsilon = 1e-4
mu = 5.0  # valor inicial à esquerda do ponto crítico
alpha = 0.01  # taxa de aprendizado

def grad(mu, y):
    return np.sum((y - mu) / (mu**3 * y)) * 2 + 0  # derivada de f(mu, y)

error = 1.0
while error > epsilon:
    mu_new = mu - alpha * grad(mu, y)
    error = abs(mu_new - mu)
    mu = mu_new

round(mu, 3)

#%%
### 14 - Método de Newton


# Dados
y = np.array([10.179, 10.073, 10.505, 10.022, 10.041,
              10.557, 10.147, 10.408, 9.785, 9.860])

# Função f(μ)
def f(mu):
    return np.sum((y - mu)**2 / (mu**2 * y)) - 12

# Derivada f'(μ)
def f_prime(mu):
    return np.sum(
        (2 * (mu - y)) / (mu**3 * y) + 
        (2 * (y - mu)**2) / (mu**4 * y)
    )

# Método de Newton
def newton_method(f, f_prime, x0, tol=1e-4, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fpx) < 1e-10:
            raise ValueError("Derivada próxima de zero.")
        x_new = x - fx / fpx
        if abs(x_new - x) < tol:
            return round(x_new, 3)
        x = x_new
    raise RuntimeError("Número máximo de iterações excedido.")

# Chute inicial à direita da média dos y
mu0 = 1.0
raiz = newton_method(f, f_prime, mu0)
print("Raiz à direita do ponto crítico:", raiz)

#%%
### 15 - Integral


lambda_ = 7982
n = 18

x, w = laggauss(n)

def integrand(u):
    t = u / 0.12
    return (lambda_**3 - t)

result = (1 / 0.12) * np.sum(w * integrand(x))

print(f'{result:.3f}')

#%%
### 16

import numpy as np


lam = -0.129

n_pontos = 21

def f(y, lam):

  return np.full_like(y, lam)

y_i, w_i = np.polynomial.hermite.hermgauss(n_pontos)

integral_valor = np.sum(w_i * f(y_i, lam))

print(f"O valor da integral é: {integral_valor:.3f}")
#%%
### 17

lam = -0.062
n_pontos = 19

y_i, w_i = laggauss(n_pontos)

fy = np.ones_like(y_i)

integral_laguerre = np.sum(w_i * fy)
resultado_final = 2 * lam * integral_laguerre

print(f"O valor da integral é: {resultado_final:.3f}")
#%%

import sympy as sp

x = sp.symbols('x', real=True)
lam = -0.062

integral_expr = lam * sp.exp(-sp.Abs(x - lam))
integral = sp.integrate(integral_expr, (x, -sp.oo, sp.oo))

resultado = integral.evalf()

print(f"Resultado simbólico: {integral}")
print(f"Resultado numérico para lambda = {lam}: {resultado:.3f}")

#%%
### 18



from numpy.polynomial.legendre import leggauss

lam = 9
n_pontos = 18
a = 0
b = 20  # limite superior para aproximar infinito

x_i, w_i = leggauss(n_pontos)
x_mapped = 0.5 * (b - a) * x_i + 0.5 * (b + a)

def integrand(x, lam):
    return lam * np.exp(-np.sqrt(x)) / np.sqrt(x)

integral = 0.5 * (b - a) * np.sum(w_i * integrand(x_mapped, lam))

print(f"O valor da integral é: {integral:.3f}")

#%%
### 19 - encontrar raizes da equação



def f(x):
    return (3*x - 4*np.sin(x)) / (6*x)

a = 1.0
b = 2.0
tol = 1e-4

while (b - a) / 2 > tol:
    c = (a + b) / 2
    if f(c) == 0:
        break
    elif f(a) * f(c) < 0:
        b = c
    else:
        a = c

raiz = (a + b) / 2
print(f"A raiz aproximada é: {raiz:.3f}")

#%%
### 21 - Integral


import numpy as np
from numpy.polynomial.hermite import hermgauss

lam = -0.033
n = 19

x, w = hermgauss(n)

# Substituindo variável para encaixar a integral:
# ∫ λ exp(-(x - λ)^2) dx = ∫ λ exp(-t^2) dt, com t = x - λ

# Portanto, f(t) = λ (constante)
f = lam * np.ones_like(x)

integral = np.sum(w * f)

print(f"{integral:.3f}")

#%%
### 22


lam = 0.172
n = 20

x, w = laggauss(n)

f = lam * np.exp(0*x)  # exp(0) = 1, pois Gauss-Laguerre já tem o peso e^{-x}

integral = np.sum(w * f)
resultado = 2 * integral

print(f"{resultado:.3f}")
#%%
### 24


lam = 16
n = 22

x, w = laggauss(n)

def integrand(t, lam):
    return lam * np.exp(-np.sqrt(t)) / np.sqrt(t)

result = np.sum(w * integrand(x, lam))
print(f"{result:.3f}")

#%%
### 25



np.random.seed(123)
mu = -0.069
n = 952
a, b = mu - 20, mu + 20
samples = np.random.uniform(a, b, n)


def integrand(x, mu):
    return 0.5 * np.exp(-np.abs(x - mu))

vals = integrand(samples, mu)

integral_estimate = (b - a) * np.mean(vals)
print(f"{integral_estimate:.3f}")

#%%
### 27 - Determinar raiz por gradiente descendente



y = np.array([0.182, 1.696, 1.621, 2.448, 0.879, 2.667, 3.886, 0.964, 0.683, 0.033])
tolerance = 0.0001
max_iter = 10000
alpha = 0.01  # taxa de aprendizado (tuning)
mu = 0.5  # chute inicial (escolha razoável > 0)

def f(mu, y):
    return -np.sum(2 * (y/mu - 1 + np.log(mu/y))) - 3.84

def grad_f(mu, y):
    n = len(y)
    # derivada de f em relação a mu
    return -2 * np.sum((-y/(mu**2)) + 1/mu)

for i in range(max_iter):
    grad = grad_f(mu, y)
    mu_new = mu - alpha * grad
    if abs(mu_new - mu) < tolerance:
        break
    mu = mu_new

print(f"Raiz encontrada: {mu:.3f}")

#%%
### 29 - Valores que optimizam a função

from scipy.optimize import minimize

xi = np.array([2, 2, 2, 3, 5, 2, 3, 1, 1, 0])
yi = np.array([-1.026, -0.582, 0.343, 0.049, 2.014, -0.334, 0.394, -1.231, 1.405, -0.667])

def obj_func(params):
    beta0, beta1 = params
    mu = beta0 + beta1 * xi
    return np.sum(np.log(np.cosh(mu - yi)))

initial_guess = [0, 0]

result = minimize(obj_func, initial_guess, method='CG')

beta0_opt, beta1_opt = result.x
obj_val = result.fun

print(f"{beta0_opt:.3f} {beta1_opt:.3f} {obj_val:.3f}")



