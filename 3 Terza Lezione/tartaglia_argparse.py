"""
Programma per calcolare il trinagolo di tartaglia
"""
import argparse
import numpy as np

description='Programma per calcolare il trinagolo di tartaglia leggendo le informazioni da linea di comando'
# descrizione accessibile con -h su shell
parser = argparse.ArgumentParser(description=description)
parser.add_argument('dim',  help='Dimensione della matrice, ovvero potenza del binomio')
args = parser.parse_args()
n = int(args.dim) # accedo alla variabile tramite il nome messo a linea 10

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