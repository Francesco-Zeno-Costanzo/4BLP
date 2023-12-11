import numpy as np

array1=np.array([1.0, 2.0, 4.0, 8.0, 16.0])

#Aggiungiamo ora un numero in una certa posizione dell'array:
array1 = np.insert(array1, 4, 18)
'''
abbiamo aggiunto il numero 18 in quarta posizione, la sintassi e' :
np.insert(array a cui vogliamo aggiungere un numero, posizione dove aggiungerlo, numero)
'''
print(array1)

#Per aggiungere elementi in fondo ad un array esiste anche il comando append della libreria numpy:
array2 = np.append(array1, -4.)
print(array2)
#Mentre per togliere un elemento basta indicare il suo indice alla funzione remove di numpy:
array2 = np.delete(array2, 0)
print(array2)