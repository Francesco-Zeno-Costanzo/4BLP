import numpy as np

#Leggiamo da un file di testo classico
path = 'dati.txt'
dati1, dati2 = np.loadtxt(path, unpack=True)
"""
unpack=True serve proprio a dire che vogliamo che
dati1 contenga la prima colonna e dati2 la seconda
La prima riga avendo il cancelletto verra' saltata
"""

#se vogliamo invece che venga letto tutto come una matrice scriviamo:
path = 'dati.txt'
dati = np.loadtxt(path, unpack=True)
#dati sara' nella fattispecie una matrice con due colonne e 6 righe


#leggere da file.csv
path = 'dati.csv'
dati1, dati2 = np.loadtxt(path,usecols=[0,1], skiprows=1, delimiter=',',unpack=True)
"""
a differenza di quanto sopra dobbiamo specificare le colonne da usare (contiamo da zero)
a causa poi dell'organizzazione dei .csv dobbiamo dire anche quante righe skippare
"""