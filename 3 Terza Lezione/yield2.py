"""
Codice esempio di utilizzo yield e confronto con range
"""

def generatore(N):
    '''
    Funzione generatore

    Parameter
    ---------
    N : int
        limite fino a cui arrviare
    '''
    n = 0
    while n < N:
        yield n
        n += 1

for val_g, val_r in zip(generatore(10), range(10)):
    print(val_g, val_r)
