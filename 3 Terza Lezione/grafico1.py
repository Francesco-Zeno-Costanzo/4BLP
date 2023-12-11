import numpy as np
import matplotlib.pyplot as plt

#Leggiamo da un file di testo classico
path = 'dati.txt'
dati1, dati2 = np.loadtxt(path, unpack=True)

plt.figure(1) #creiamo la figura

#titolo
plt.title('Grafico dati')
#nomi degli assi
plt.xlabel('t[s]')
plt.ylabel('x[m]')
#plot dei dati
plt.plot(dati1,dati2, marker='.',linestyle='')
#aggiungiamo una griglia
plt.grid()
#comando per mostrare a schermo il grafico
plt.show()