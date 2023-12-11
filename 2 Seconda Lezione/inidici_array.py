import numpy as np

array1 = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

primi_tre = array1[0:3]
print('primi_tre = ', primi_tre)
"""
Questa sintassi seleziona gli elementi di array1
dall'indice 0 incluso all'indice 3 escluso.
Il risultato e' ancora un array.
"""

esempio = array1[1:-1]
print(esempio)
esempio = array1[-2:5]
print(esempio)
#Questo metodo accetta anche valori negativi, con effetti curiosi


elementi_pari = array1[0::2]
print('elementi_pari = ', elementi_pari)
"""
In questo esempio invece, usando invece due volte il simbolo :
intendiamo prendere solo gli elementi dall'indice 0 saltando di 2 in 2.
Il risultato e' un array dei soli elementi di indice pari
"""

rewind = array1[::-1]
print('rewind = ', rewind)
"""
Anche qui possiamo usare valori negativi.
In particolare questo ci permette di saltare "all'indietro"
e, ad esempio, di invertire l'ordine di un'array con un solo comando
"""