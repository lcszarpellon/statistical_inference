### 01 - Calcular dy/dx
import sympy as sp

# Define a variável simbólica
x = sp.symbols('x')

# Define u e y em termos de x
u = 3 * x**4 + 5
y = u**6

# Calcula a derivada de y em relação a x
dy_dx = sp.diff(y, x)

# Mostra o resultado simplificado
print("dy/dx =", dy_dx)
#%%
### 02 - Calcule a Matriz C = A - B

import numpy as np

# Define a matriz A
A = np.array([
    [4, 10, 13, 8, 16],
    [11, 9, 7, 8, 8],
    [9, 14, 8, 10, 12],
    [15, 13, 10, 8, 11],
    [5, 10, 7, 2, 12]
])

# Define a matriz B
B = np.array([
    [8, 6, 9, 10, 13],
    [8, 11, 11, 16, 13],
    [5, 14, 16, 18, 8],
    [5, 11, 13, 10, 6],
    [9, 8, 6, 7, 6]
])

C = A - B

# Soma todos os elementos de C
soma_total = np.sum(C)

print(f"Soma total dos elementos de C: {soma_total:.3f}")

#%%
### 03 - Desenhar e interpretar o gráfico de uma função
import matplotlib.pyplot as plt

# Define a função
def f(x):
    return x**2 + 2

# Gera um intervalo de x entre -10 e 10
x = np.linspace(-10, 10, 400)
y = f(x)

# Plota o gráfico
plt.plot(x, y, label='f(x) = x² + 2', color='blue')
plt.title('Gráfico de f(x) = x² + 2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

#%%
### 05 - Gerando gráfico 2D e 3D a partir de uma função

def f(x, y):
    return 1 - x - 0.5 * y

# Cria a malha de pontos no intervalo [-2, 2]
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Cria os gráficos lado a lado
fig = plt.figure(figsize=(12, 5))

# Gráfico 3D (perspectiva)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='gray', edgecolor='black')
ax1.set_title('Gráfico de Perspectiva')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('f(x, y)')

# Gráfico de contorno (isolinha)
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z, cmap='YlOrBr', levels=20)
ax2.contour(X, Y, Z, colors='black', linewidths=0.5)
ax2.set_title('Gráfico de Isolinhas')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
fig.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.show()

#%%
### 06 - Obter o log do determinante da inversa da matriz A


# Autovalores fornecidos
auto_val_A = np.array([29.7, 2.7, 2.7, 0.9])

# Log do determinante da inversa: - soma dos logs dos autovalores
log_det_inv = -np.sum(np.log(auto_val_A))

print(f"Log do determinante da Inversa de A {log_det_inv:.4f}")

#%%
### 08 - Resolver Integral

np.random.seed(123)

# Número de pontos de integração
n = 919

mu = -0.062

# Laplace
def laplace_pdf(x, mu):
    return 0.5 * np.exp(-np.abs(x - mu))


# Amostragem Monte Carlo:
samples = np.random.laplace(loc=mu, scale=1, size=n)

values = laplace_pdf(samples, mu)

# Integral estimada como média dos valores (Monte Carlo)
integral_estimate = np.mean(values) * (samples.max() - samples.min())

print(f"Estimativa da integral via Monte Carlo: {integral_estimate:.3f}")

#%%
### 10 - Encontrar raiz da equação pelo método bisseção

import math

def f(x):
    return x**2 - math.exp(-x)

def bissecao(a, b, tol):
    if f(a) * f(b) >= 0:
        print("Função não muda de sinal no intervalo dado")
        return None

    xi_old = a
    xi = (a + b) / 2.0

    while abs(xi - xi_old) >= tol:
        if f(a) * f(xi) < 0:
            b = xi
        else:
            a = xi

        xi_old = xi
        xi = (a + b) / 2.0

    return xi

# Parâmetros
a = 0
b = 1
tol = 0.0001

raiz = bissecao(a, b, tol)
print(f"Raiz aproximada: {raiz:.3f}")

#%%
### 11 - Obter valores singulares da matriz A

# Definindo a matriz A
A = np.array([
    [11, 9, 9, 12, 8],
    [8, 8, 12, 6, 11],
    [6, 11, 8, 10, 7],
    [12, 11, 11, 12, 15]
])

# Calculando os valores singulares
singular_values = np.linalg.svd(A, compute_uv=False)


for i, val in enumerate(singular_values, start=1):
    print(f"Valor singular {i}: {val:.3f}")

#%%
### 12 - Otimização de função

from scipy.optimize import minimize

# Dados fornecidos
x = np.array([6, 0, 1, 1, 3, 1, 3, 4, 2, 2])
y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])
n = len(y)

# Função objetivo: log-verossimilhança negativa média
def L(beta):
    beta0, beta1 = beta
    linear_pred = beta0 + beta1 * x
    mu = np.exp(linear_pred) / (1 + np.exp(linear_pred))
    # Evitar log(0) com epsilon
    epsilon = 1e-10
    mu = np.clip(mu, epsilon, 1 - epsilon)
    equacao_L = y * np.log(mu) + (1 - y) * np.log(1 - mu)
    return -np.sum(equacao_L) / n

# Valores iniciais beta0=0, beta1=0
beta_init = np.array([0.0, 0.0])

# Minimização usando BFGS
result = minimize(L, beta_init, method='BFGS')

beta0_opt, beta1_opt = result.x
L_opt = result.fun

print(f"β0 = {beta0_opt:.3f}")
print(f"β1 = {beta1_opt:.3f}")
print(f"Valor da função objetivo no ponto ótimo = {L_opt:.3f}")

#%%
### 13 - Resolvendo integral por Gauss-Legendre

from numpy.polynomial.legendre import leggauss

# Parâmetro lambda
lam = 0.016

# Função a integrar
def f(x):
    return lam * np.exp(-np.abs(x - lam))

# Intervalo 
a = lam - 10
b = lam + 10

# Número de pontos
n = 20

# Nós e pesos da quadratura Gauss-Legendre
nodes, weights = leggauss(n)

# Transformação dos nós para o intervalo [a, b]
x = 0.5 * (nodes + 1) * (b - a) + a
w = 0.5 * (b - a) * weights

# Estimativa da integral
integral = np.sum(w * f(x))

print(f"Integral estimada por Gaussiana: {integral:.3f}")

#%%
### 14 - Resolvendo integral por Gauss-Hermite

from numpy.polynomial.hermite import hermgauss

# Valor de lambda
lam = 0.091

# Número de pontos
n = 21

# Obter pontos e pesos de Gauss-Hermite
x, w = hermgauss(n)

# Gauss-Hermite -> f(x) e^{-x^2}, então g(x) = f(x + lambda)
# Função dada -> lambda * exp(-(x - lambda)^2) = lambda * exp(-y^2),
# Com y = x - lambda -> lambda * exp(-y^2).

# é fácil ver que: lambda * e^{-(x - lambda)^2} -> y = x - lambda:
# integral = \int f(y) e^{-y^2} dy = \int \lambda dy = infinito,
# dado f(y) = lambda, então
# A integral é lambda * sqrt(pi)

integral = lam * np.sum(w)

print(f"Integral estimada por quadratura Gauss-Hermite: {integral:.3f}")

#%%
### 15 - Encontrando a raiz positiva de um polinômio usando método da bisseção

def f(x):
    return x**3 + 3.8 * x**2 - 8.6 * x - 24.4

def bissecao(a, b, tol):
    if f(a) * f(b) >= 0:
        raise ValueError("A função não muda de sinal no intervalo dado.")

    xi_old = a
    xi = (a + b) / 2.0

    while abs(xi - xi_old) >= tol:
        if f(a) * f(xi) < 0:
            b = xi
        else:
            a = xi

        xi_old = xi
        xi = (a + b) / 2.0

    return xi

# Intervalo inicial
a = 0
b = 3
tol = 0.0001

raiz = bissecao(a, b, tol)
print(f"Raiz positiva aproximada: {raiz:.3f}")

#%%
### 16 - Soma de vetores

a = np.array([13, 11, 4, 10, 12])
b = np.array([11, 12, 10, 12, 15])

# Soma elemento a elemento
c = a + b

# Soma dos elementos de c
resultado = np.sum(c)

print(f"Soma total dos elementos de c: {resultado:.3f}")

#%%
### 17 - Produtos de vetores

a = np.array([12, 8, 7, 5, 3])
b = np.array([10, 9, 14, 15, 6])

# Calculando o produto interno
produto_interno = np.dot(a, b)

print(f"Produto interno de a por b: {produto_interno:.3f}")

#%%
### 19 - Resolução de um sistema de equações não linear usando método de Newton

def F(xy):
    x, y = xy
    f1 = x + y - x*y + 2
    f2 = x * np.exp(-y) - 1
    return np.array([f1, f2])

def J(xy):
    x, y = xy
    df1_dx = 1 - y
    df1_dy = 1 - x
    df2_dx = np.exp(-y)
    df2_dy = -x * np.exp(-y)
    return np.array([[df1_dx, df1_dy],
                     [df2_dx, df2_dy]])

def newton_method(x0, tol=1e-4, max_iter=100):
    x_old = np.array(x0, dtype=float)
    for _ in range(max_iter):
        fx = F(x_old)
        Jx = J(x_old)
        delta = np.linalg.solve(Jx, fx)
        x_new = x_old - delta
        if np.linalg.norm(x_new - x_old, ord=np.inf) < tol:
            return x_new
        x_old = x_new
    raise ValueError("Não convergiu dentro do número máximo de iterações")

# Escolha do valor inicial 
x0 = [0, 0]

raiz = newton_method(x0)

print(f"x̂ = {raiz[0]:.3f}")
print(f"ŷ = {raiz[1]:.3f}")

#%%
### 20 - Obter o log do determinante da inversa da matriz A

# Autovalores fornecidos
auto_val_A_20 = np.array([30.6, 1.8, 1.8, 1.8])

# Log do determinante da inversa: - soma dos logs dos autovalores
log_det_inv_20 = -np.sum(np.log(auto_val_A))

print(f"Q20 - Log do determinante da Inversa de A {log_det_inv:.3f}")

#%%
### 21 - Produto matricial

A = np.array([
    [9, 8, 6, 11, 3],
    [4, 13, 16, 9, 6],
    [6, 12, 16, 11, 5],
    [7, 8, 6, 12, 12],
    [12, 6, 7, 9, 13]
])

B = np.array([
    [3, 10, 8, 13, 7],
    [11, 7, 14, 13, 7],
    [8, 8, 11, 13, 8],
    [8, 3, 14, 10, 10],
    [5, 12, 8, 13, 13]
])

# Calculando o produto matricial
C = np.dot(A, B)

# Somando todos os elementos da matriz C
resultado = np.sum(C)

print(f"Soma de todos os elementos de C: {resultado:.3f}")

#%%
### 25 - Resolvendo a distribuição de Gamma por Gauss-Legendre

from math import gamma

a = 0.1
b = 11.1
alpha = 2
beta = 3

def integrand(y):
    coef = (beta ** alpha) / gamma(alpha)
    return coef * (y ** (alpha - 1)) * np.exp(-beta * y)

n = 22
nodes, weights = leggauss(n)

x = 0.5 * (nodes + 1) * (b - a) + a
w = 0.5 * (b - a) * weights

# Calcular a integral
integral = np.sum(w * integrand(x))

print(f"Integral estimada por Gauss-Legendre: {integral:.3f}")

#%%
### 25 - Produto de Hadamard

a = np.array([8, 11, 15, 5, 10])
b = np.array([9, 8, 12, 15, 9])

# Produto de Hadamard (elemento a elemento)
c = a * b

# Soma dos elementos de c
resultado = np.sum(c)

print(f"Soma do produto de Hadamard: {resultado:.3f}")

#%%
### 28 - Angulo entre dois vetores

a = np.array([9, 9, 6, 8, 11])
b = np.array([4, 8, 18, 6, 14])

# Produto interno
dot_product = np.dot(a, b)

# Normas dos vetores
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)

# Cálculo do cosseno do ângulo
cos_theta = dot_product / (norm_a * norm_b)

# Garantir que o valor fique no intervalo válido [-1,1] para arccos
cos_theta = np.clip(cos_theta, -1.0, 1.0)

# angulo em radianos
theta_rad = np.arccos(cos_theta)
# angulo em graus
theta_deg = np.degrees(theta_rad)

print(f"Ângulo theta entre a e b (em radianos): {theta_rad:.3f}")
print(f"Ângulo theta entre a e b (em graus): {theta_deg:.3f}")

#%%
### 29 - Calculo de derivadas

# Definir variável simbólica
x = sp.Symbol('x')

# Definir a função
y = (5 * x) / (1 - 3 * x)

# Calcular derivadas
dy_dx = sp.diff(y, x)
d2y_dx2 = sp.diff(dy_dx, x)

# Avaliar em x = 2
valor_primeira = dy_dx.subs(x, 2).evalf()
valor_segunda = d2y_dx2.subs(x, 2).evalf()

print(f"Primeira derivada em x=2: {valor_primeira:.3f}")
print(f"Segunda derivada em x=2: {valor_segunda:.3f}")

#%%
### 30 - Obtendo valores singulares de uma matriz

# Definindo a matriz A
A = np.array([
    [17,  9,  9,  9,  7],
    [14, 10, 16, 11,  9],
    [ 6, 12, 16, 10, 10],
    [ 7, 14, 17, 13,  5]
])

# Calculando os valores singulares
U, S, Vt = np.linalg.svd(A)

# Imprimindo os quatro primeiros valores singulares
for i, s in enumerate(S[:4], start=1):
    print(f"Valor singular {i}: {s:.3f}")

