# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 08:49:35 2025

@author: lucas
"""

### 03 -

from scipy.stats import binom

# Parâmetros
n = 10
p = 0.35
k = 7

# Calculando P(X = 7)
prob = binom.pmf(k, n, p)
print(f'P(X=7) = {prob:.2f}')

#%%
### 05 - Poisson

from scipy.stats import poisson

# Parâmetros
lambda_ = 10
k = 7

# Calculando P(X = 7)
prob = poisson.pmf(k, mu=lambda_)
print(f'P(X = 7) = {prob:.2f}')

#%%
### 06

lambda_ = 150
k = 150

# P(X > 150) = 1 - P(X <= 150)
prob = 1 - poisson.cdf(k, mu=lambda_)
print(f'P(X > 150) = {prob:.2f}')

#%%
### 08 - Distrubuição normal

from scipy.stats import norm

# Parâmetros
mu = 100
sigma = 10
x = 95

# Probabilidade acumulada até x = 95
prob = norm.cdf(x, loc=mu, scale=sigma)
print(f'P(X < 95) = {prob:.2f}')

#%%
### 17 

mu = 2
sigma = 5**0.5
lower = 1
upper = 5

prob = norm.cdf(upper, loc=mu, scale=sigma) - norm.cdf(lower, loc=mu, scale=sigma)
print(f'P(1 < X < 5) = {prob:.2f}')



