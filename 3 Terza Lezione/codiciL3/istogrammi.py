import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
plt.title('grafico a barre')
plt.xlabel('valore')
plt.ylabel('conteggi')
# Sull'asse x utilizziamo un array di 10 punti equispaziati.
x = np.linspace(1,10,10)
# Sull'asse y abbiamo, ad esempio, il seguente set di dati:
y = np.array([2.54, 4.78, 1.13, 3.68, 5.79, 7.80, 5.4, 3.7, 9.0, 6.6])

# Il comando per la creazione dell'istogramma corrispondente e':
plt.bar(x, y, align='center')

plt.figure(2)
plt.title('istogramma di una distribuzione gaussiana')
plt.xlabel('x')
plt.ylabel('p(x)')

"""
lista di numeri distribuiti gaussianamente con media 0 e varianza 1
si usa l'underscore nel for poiche' non serve usare
un'altra variabile. Avremmo potuto scrivere for i ...
ma la i non sarebbe comparsa da nessun' altra parte
sarebbe stato uno spreco
"""
z = [np.random.normal(0, 1) for _ in range(int(1e5))]
plt.hist(z, bins=50, density=True, histtype='step')
plt.minorticks_on() # tick più piccoli sugli assi

plt.show()