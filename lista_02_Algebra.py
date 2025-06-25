# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 19:25:43 2025

@author: lucas
"""
### 01 - Autovalores de uma matriz

import numpy as np

# Definindo a matriz A
A = np.array([
    [2.0, 1.6, 1.6, 1.6],
    [1.6, 2.0, 1.6, 1.6],
    [1.6, 1.6, 2.0, 1.6],
    [1.6, 1.6, 1.6, 2.0]
])

# Calculando os autovalores
autovalores = np.linalg.eigvals(A)

# Imprimindo os autovalores com 3 casas decimais
for i, val in enumerate(sorted(autovalores, reverse=True), start=1):
    print(f"Autovalor {i}: {val:.3f}")

#%%
### 02 - Produto escalar

a = np.array([10, 11, 7, 13, 18])
alpha = 10

b = alpha * a

resultado = np.sum(b)

print(f"Soma de b = alpha*a: {resultado:.3f}")

#%%
### 03 - Sistema linear com Gauss-Jordan

def gauss_jordan(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

 
    aug = np.hstack([A, b.reshape(-1, 1)])

    for i in range(n):
        if aug[i, i] == 0:
            for j in range(i+1, n):
                if aug[j, i] != 0:
                    aug[[i, j]] = aug[[j, i]]
                    break
        aug[i] = aug[i] / aug[i, i]
  
        for j in range(n):
            if j != i:
                factor = aug[j, i]
                aug[j] = aug[j] - factor * aug[i]

    return aug[:, -1]

A = np.array([
    [2.0, 0.8, 0.8, 0.7],
    [0.8, 2.0, 0.7, 0.8],
    [0.8, 0.7, 2.0, 0.8],
    [0.7, 0.8, 0.8, 2.0]
])

b = np.array([8, 12, 11, 10])

solucao = gauss_jordan(A, b)

for i, xi in enumerate(solucao, start=1):
    print(f"x{i} = {xi:.3f}")

#%%
### 04 - Derivada da matriz

tau0 = 2
tau1 = 1


I = np.eye(4)

Z = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
])


A = tau0 * I + tau1 * Z


A_inv = np.linalg.inv(A)

resultado = np.trace(A_inv @ Z)

print(f"{resultado:.3f}")

#%%
### 05 - Determinante e traço da inversa


A = np.array([
    [1.00, 0.78, 0.78, 0.71],
    [0.78, 1.00, 0.71, 0.78],
    [0.78, 0.71, 1.00, 0.78],
    [0.71, 0.78, 0.78, 1.00]
])


det_A = np.linalg.det(A)

A_inv = np.linalg.inv(A)


trace_A_inv = np.trace(A_inv)

print(f"Determinante de A: {det_A:.3f}")
print(f"Traço da inversa de A: {trace_A_inv:.3f}")

#%%
### 06 - solução de sitemas

from scipy.linalg import lu, solve_triangular

A = np.array([
    [1.0, 0.9, 0.9, 0.8],
    [0.9, 1.0, 0.8, 0.9],
    [0.9, 0.8, 1.0, 0.9],
    [0.8, 0.9, 0.9, 1.0]
])

b = np.array([2, 10, 5, 7])

P, L, U = lu(A)

# Resolver Ly = Pb
Pb = P @ b
y = solve_triangular(L, Pb, lower=True)

soma_y = np.sum(y)

print(f"Soma do vetor y (intermediário): {soma_y:.3f}")
#%%
### 07 - Traço da matriz

from scipy.linalg import lu


A = np.array([
    [1.0, 0.8, 0.8, 0.7],
    [0.8, 1.0, 0.7, 0.8],
    [0.8, 0.7, 1.0, 0.8],
    [0.7, 0.8, 0.8, 1.0]
])

# Decomposição LU
P, L, U = lu(A)

# Traço de U
trace_U = np.trace(U)

print(f"Traço da matriz U: {trace_U:.3f}")

#%%
### 09 - resolução de sistemas por Jacobi


def jacobi(A, b, tol=1e-4, max_iter=1000):
    A = np.array(A)
    b = np.array(b)
    n = len(b)
    x_old = np.zeros(n)
    x_new = np.zeros(n)

    for iteration in range(1, max_iter + 1):
        for i in range(n):
            s = sum(A[i][j] * x_old[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        if np.linalg.norm(x_new - x_old, ord=np.inf) < tol:
            return iteration

        x_old[:] = x_new[:]

    return max_iter  


A = [
    [2.5, 0.9, 0.9, 0.8],
    [0.9, 2.5, 0.8, 0.9],
    [0.9, 0.8, 2.5, 0.9],
    [0.8, 0.9, 0.9, 2.5]
]

b = [12, 12, 6, 20]

# Executar método de Jacobi
n_iter = jacobi(A, b)

print(f"Número de iterações até convergir: {n_iter}")

#%%
### 10 - Sitema linear por Gauss-Jordan

import numpy as np

def gauss_jordan(A, b, tol=1e-10):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)

    # Construir a matriz aumentada [A | b]
    aug = np.hstack([A, b.reshape(-1, 1)])

    for i in range(n):
        # Pivotamento parcial (se necessário)
        max_row = np.argmax(abs(aug[i:, i])) + i
        if abs(aug[max_row, i]) < tol:
            raise ValueError("Matriz singular ou quase singular.")
        if i != max_row:
            aug[[i, max_row]] = aug[[max_row, i]]

        # Tornar o pivô igual a 1
        aug[i] = aug[i] / aug[i, i]

        # Eliminar as outras linhas
        for j in range(n):
            if j != i:
                aug[j] -= aug[j, i] * aug[i]

    # Solução é a última coluna da matriz aumentada
    return aug[:, -1]

# Dados
A = [
    [2.0, 0.8, 0.8, 0.8],
    [0.8, 2.0, 0.8, 0.8],
    [0.8, 0.8, 2.0, 0.8],
    [0.8, 0.8, 0.8, 2.0]
]

b = [13, 15, 12, 8]

# Resolver o sistema
x = gauss_jordan(A, b)

# Exibir resultado com 3 casas decimais
print("Solução [x1, x2, x3, x4]:")
print([round(xi, 3) for xi in x])

#%%
### 11

B = [
    [2.0, 0.8, 0.8, 0.7],
    [0.8, 2.0, 0.7, 0.8],
    [0.8, 0.7, 2.0, 0.8],
    [0.7, 0.8, 0.8, 2.0]
]

c = [18, 11, 12, 8]

# Resolver o sistema
x = gauss_jordan(B, c)

# Exibir resultado com 3 casas decimais
print("Solução [x1, x2, x3, x4]:")
print([round(xi, 3) for xi in x])

#%%
### 13 - Subtração vetorial

a = np.array([7, 11, 13, 7, 11])
b = np.array([7, 6, 8, 11, 10])

# Subtração
c = a - b

# Soma dos elementos de c
resultado = np.sum(c)

print(f"Soma de c = {resultado:.3f}")

#%%
### 14 - Exponencial e traço da matriz


from scipy.linalg import expm


A = np.array([
    [1.0, 0.8, 0.8, 0.7],
    [0.8, 1.0, 0.7, 0.8],
    [0.8, 0.7, 1.0, 0.8],
    [0.7, 0.8, 0.8, 1.0]
])


exp_A = expm(A)

trace_exp_A = np.trace(exp_A)

print(f"Traço da exponencial matricial de A: {trace_exp_A:.3f}")

#%%
### 15 - Sistema por eliminação de Gauss


def gauss_elimination_no_pivot(A, b, tol=1e-10):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

  
    for k in range(n-1):
        for i in range(k+1, n):
            if abs(A[k, k]) < tol:
                raise ValueError("Divisão por zero ou pivô muito pequeno.")
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]


    x = np.zeros(n)
    for i in reversed(range(n)):
        soma = np.dot(A[i, i+1:], x[i+1:])
        x[i] = (b[i] - soma) / A[i, i]

    return x


A = np.array([
    [1.0, 0.5, 0.5, 0.4],
    [0.5, 1.0, 0.4, 0.5],
    [0.5, 0.4, 1.0, 0.5],
    [0.4, 0.5, 0.5, 1.0]
])

b = np.array([5, 8, 11, 9])


x = gauss_elimination_no_pivot(A, b)

print("Solução [x1, x2, x3, x4]:")
print([round(xi, 3) for xi in x])

#%%
### 16 - Decomposição matricial

from scipy.linalg import lu

A = np.array([
    [1.0, 0.9, 0.9, 0.9],
    [0.9, 1.0, 0.9, 0.9],
    [0.9, 0.9, 1.0, 0.9],
    [0.9, 0.9, 0.9, 1.0]
])

b = np.array([12, 5, 15, 7])

P, L, U = lu(A)
y = np.linalg.solve(L, np.dot(P, b))
print(round(np.sum(y), 3))

#%%
### 17 - Sistema usando Gauss-Seidel


A = np.array([
    [2.5, 0.8, 0.8, 0.8],
    [0.8, 2.5, 0.8, 0.8],
    [0.8, 0.8, 2.5, 0.8],
    [0.8, 0.8, 0.8, 2.5]
])
b = np.array([11, 5, 11, 11])

def gauss_seidel(A, b, tol=1e-4, max_iter=1000):
    x = np.zeros(len(b))
    n = len(b)
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return it
    return it

iteracoes = gauss_seidel(A, b)
print(iteracoes)

#%%
### 18 - Traço da matriz

A = np.array([
    [1.0, 0.9, 0.9, 0.9],
    [0.9, 1.0, 0.9, 0.9],
    [0.9, 0.9, 1.0, 0.9],
    [0.9, 0.9, 0.9, 1.0]
])

a, b, U = lu(A)
trace_U = np.trace(U)
print(round(trace_U, 3))

#%%
### 19 - Soma matricial


A = np.array([
    [13, 9, 13, 12, 10],
    [11, 7, 12, 9, 18],
    [14, 11, 8, 8, 8],
    [18, 9, 15, 17, 14],
    [5, 11, 9, 11, 4]
])

B = np.array([
    [10, 13, 17, 15, 11],
    [9, 11, 7, 6, 10],
    [13, 16, 8, 5, 11],
    [7, 9, 10, 9, 9],
    [14, 16, 12, 10, 8]
])

C = A + B
result = np.sum(C)
print(round(result, 3))

#%%
### 20 - Valores singulares

A = np.array([
    [8, 6, 10, 8, 8],
    [10, 9, 6, 12, 13],
    [9, 8, 5, 11, 11],
    [8, 5, 6, 8, 7]
])

U, s, VT = np.linalg.svd(A, full_matrices=False)
singular_values = np.round(s[:4], 3)
print(singular_values)
#%%
### 21 - Ângulo entre vetores

a = np.array([11, 8, 10, 9, 7])
b = np.array([4, 15, 13, 9, 10])

cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # angle in radians
theta_deg = np.degrees(theta)
print(round(theta_deg, 3))

#%%
### 22 - Soma matricial

A = np.array([
    [10, 10, 10, 9, 7],
    [8, 9, 11, 5, 11],
    [11, 16, 8, 10, 9],
    [5, 7, 4, 6, 14],
    [10, 6, 9, 10, 10]
])

B = np.array([
    [3, 7, 8, 8, 11],
    [9, 9, 14, 11, 15],
    [10, 13, 9, 6, 6],
    [9, 6, 11, 7, 12],
    [12, 5, 11, 9, 11]
])

C = A + B
result = np.sum(C)
print(round(result, 3))

#%%
### 25 - log do determinante da inversa 

eigenvalues = np.array([4.68e+01, 2.6, 2.6, 3.170522e-15])

filtered_eigenvalues = eigenvalues[eigenvalues > 1e-12]

log_det_inv = -np.sum(np.log(eigenvalues))
print(f"Log do determinante da inversa: {log_det_inv:.3f}")

#%%
### 26 - Determinante e traço


A = np.array([
    [12, 8, 13, 10, 14],
    [13, 11, 11, 10, 7],
    [4, 9, 12, 12, 11],
    [13, 7, 10, 11, 9],
    [7, 9, 16, 12, 6]
])

determinant = np.linalg.det(A)
trace = np.trace(A)

print(f"{determinant:.3f}, {trace:.3f}")

#%%
### 27 - Angulo entre vetores


a = np.array([9, 9, 6, 8, 11])
b = np.array([4, 8, 18, 6, 14])

dot_product = np.dot(a, b)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)

cos_theta = dot_product / (norm_a * norm_b)
theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # para evitar erros numéricos
theta_degrees = np.degrees(theta)

print(f"{theta_degrees:.3f}")

#%%
### 28 - Valores singulares

A = np.array([
    [17, 9, 9, 9, 7],
    [14, 10, 16, 11, 9],
    [6, 12, 16, 10, 10],
    [7, 14, 17, 13, 5]
])

U, s, VT = np.linalg.svd(A, full_matrices=False)
singular_values = s[:4]

print([round(val, 3) for val in singular_values])

#%%
### 29 - Angulo entre vetores

a = np.array([10, 12, 12, 5, 8])
b = np.array([10, 15, 10, 7, 15])

cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) 
theta_degrees = np.degrees(theta)

print(round(theta_degrees, 3))

#%%
### 30 - Traço da matriz

A = np.array([
    [1.0, 0.8, 0.8, 0.7],
    [0.8, 1.0, 0.7, 0.8],
    [0.8, 0.7, 1.0, 0.8],
    [0.7, 0.8, 0.8, 1.0]
])

P, L, U = lu(A)
trace_U = np.trace(U)
print(round(trace_U, 3))

