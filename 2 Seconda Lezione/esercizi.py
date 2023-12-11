import numpy as np

#=============================
# Primo Esercizio
#=============================
print("Primo esercizio:")

a = [1, 2, 3, 4]
print(f"array iniziale: {a}, id: {id(a)}")

b = a
b[0] = 7

print(f"array iniziale: {a}, id: {id(a)}")
print(f"array finale  : {b}, id: {id(b)}")

print()
#=============================
# Secondo Esercizio
#=============================
print("Secondo esercizio:")

I = np.zeros((3, 3))
idx = [0, 1, 2] # lista degli indici

I[idx, idx] = 1

v = np.array([4, 8, 2]) # vettore a caso
# calcolo prodotto scalare
print(v.T @ I @ v)
print(sum(v**2))

print()
#=============================
# Terzo Esercizio
#=============================
print("Terzo esercizio:")

L = np.zeros((4, 4))
idx = [0, 1, 2, 3] # lista degli indici

L[idx, idx] = [-1, 1, 1, 1]

v = np.array([4, 8, 2, 1]) # vettore a caso
print(v.T @ L @ v)
v = np.array([9, 0, 2, 1]) # vettore a caso
print(v.T @ L @ v)

print()
#=============================
# Quarto Esercizio
#=============================
print("Quarto esercizio:")

A = np.zeros((4, 2))
A[:, 0] = 1
print(A)
# faccio lo swap cambiando l'ordine degli indici
print(A[:, [1, 0]])

print()
#=============================
# Quinto Esercizio
#=============================
print("Quinto esercizio:")

x = np.linspace(0, 20, 21, dtype=int)

pari = x[x%2 == 0] # maschera per i pari
disp = x[x%2 == 1] # maschera per i dispari
print(pari)
print(disp)

print()
#=============================
# Sesto Esercizio
#=============================
print("Sesto esercizio:")

x = np.linspace(0, 20, 21, dtype=int)

# maschera con due condizioni
mask = (x > 5) & (x < 15)
# "&" per and mentre "|" per or
print(x[mask])

print()
#=============================
# Settimo Esercizio
#=============================
print("Settimo esercizio:")

x = np.linspace(0, 10, 11, dtype=int)
# trasformo il vettore in una matrice N x 1
y = x[:, None]
print(x)
print(y)
print(x*y)

