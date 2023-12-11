"""
Programma per calcolare il trinagolo di tartaglia
"""
import numpy as np

# leggo da input un valore e lo rendo intero
# la stringa verra stampata su shell
n = int(input("Ordine del triangolo: "))
a = np.zeros((n, n), dtype=int) # matrice per i coefficienti

# calocolo i coefficienti del trinagolo
a[0,0] = 1
for i in range(1, n):
    a[i, 0] = 1
    for j in range(1, i):
        a[i, j] = a[i-1, j-1] + a[i-1, j]
    a[i,i] = 1

# stampo a schermo
for i in range(n):
    for j in range(i+1):
        # solo per fare la forma a piramide
        if j == 0 : # non funziona con numeri a due cifre
            print(*[""]*(n-i),a[i, j], end='')
        else:
            print("",a[i, j], end='')
    print()