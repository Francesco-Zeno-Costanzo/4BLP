import numpy as np

def trova_pari(array):
    """
    restituisce un array contenente solo
    i numeri pari dell'array di partenza
    """
    R = np.array([]) #array da riempire
    #per ogni elemento in arrary fai ...
    for elem in array:
        if elem%2 == 0:
            R = np.append(R,elem)
    return R

a = np.array([i for i in range(0, 11)])
"""
il precedente e' un modo piu' conciso di scrivere:
a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
"""
print(a)
print(trova_pari(a))