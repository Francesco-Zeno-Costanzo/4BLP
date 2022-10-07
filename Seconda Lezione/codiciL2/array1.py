import numpy as np

#Creiamo un array di 5 elementi
array1 = np.array([1.0, 2.0, 4.0, 8.0, 16.0]) #scrivere 2.0 equivale a scrivere 2.

print(array1)

#per accedere a un singolo elemento dell'array basta fare come segue:
elem = array1[1]

#ATTENZIONE! Gli indici, per Python, partono da 0, non da 1!
print(elem)