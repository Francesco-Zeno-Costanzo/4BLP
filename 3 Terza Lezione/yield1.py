"""
Codice esempio di utilizzo yield per un generatore
"""

def generatore():
    ''' funzione generatore
    '''
    yield 1
    yield 2
    yield 3

gen = generatore()

for val in gen:
    print(val)
