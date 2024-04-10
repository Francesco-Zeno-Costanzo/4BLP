"""
Codice per verificare il consumo di ram
"""

import sys

N = int(2e8) # quanti numeri generare 2 x 10**8

def gen(N):
    '''
    Funzione generatore

    Parameter
    ---------
    N : int
        limite fino a cui arrviare
    '''
    for i in range(N):
        yield i

# Conservo una tutto in una lista
n_l = list(gen(N))

# Definisco semplicemente il generatore
n_g = gen(N)

def size(obj):
    '''
    Funzione per calcolare tutta la memoria di un ogetto

    Parameter
    ---------
    obj : list, tuple, set, dict
        python object
    '''
    Size = sys.getsizeof(obj)

    # se è una lista tupla o set
    if isinstance(obj, (list,tuple,set)):
        for el in obj:
            Size += size(el)

    # se è un dizionario devo considerare sia chiave che valore
    if isinstance(obj, dict):
        for k, v in obj.items():
            Size += size(k)
            Size += size(v)

    return Size

print(f"Dimensione della lista:    {size(n_l)} byte")

del n_l # elimino la variabile dalla memoria in quanto incredibilmente pesante

print(f"Dimensione del generatore: {size(n_g)} byte")
