import numpy as np

# creo un array
x = np.array([5, 4, 2, 8, 3, 9, 7, 2, 6, 3, 9, 8])

# voglio selezionare solo gli elementi maggiori di una certa soglia
mask = x >= 4 # mask è un array di booleani, secondo la condizione data
# mask vale True negli indici in cui il valore di x e' maggiore o uguale a 4

print(x)
print(mask)
print(x[mask])
