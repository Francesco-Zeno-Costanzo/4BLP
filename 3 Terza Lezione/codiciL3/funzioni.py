def area(a, b):
    """
    restituisce l'area del rettangolo
    di lati a e b
    """
    A = a*b #calcolo dell'area
    return A

#chiamiamo la funzione e stampiamo subito il risultato
print(area(3, 4))
print(area(2, 5))

"""
Se la funzione non restituisce nulla
ma esegue solo un pezzo di codice,
si parla propriamente di procedura
e il valore restituito e' None.
"""
def procedura(a):
    a = a+1

print(procedura(2))

"""
Volendo si possono creare anche funzioni
che non hanno valori in ingresso:
"""
def pigreco():
    return 3.14
print(pigreco())