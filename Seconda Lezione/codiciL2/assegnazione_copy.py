import numpy as np

a = np.array([1, 2, 3, 4])
print(f"array iniziale: {a}, id: {id(a)}")

b = a
b[0] = 7

print(f"array iniziale: {a}, id: {id(a)}")
print(f"array finale  : {b}, id: {id(b)}")

#usiamo ora copy invece che l'assegnazione

a = np.array([1, 2, 3, 4])
print(f"array iniziale: {a}, id: {id(a)}")

b = np.copy(a)
b[0] = 7

print(f"array iniziale: {a}, id: {id(a)}")
print(f"array finale  : {b}, id: {id(b)}")