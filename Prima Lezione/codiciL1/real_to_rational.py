#variabili
x = 0.1
y = 0.2
z = 0.3
#sommo le prime due
t = x + y

"""
applico una funzione che mi fornisce
una tupla contenente due numeri interi
il cui rapporto restituisce il numero iniziale.
output del tipo: (numeratore, denominatore)
"""
print(t.as_integer_ratio())
print(z.as_integer_ratio())

[Output]
(1351079888211149, 4503599627370496)
(5404319552844595, 18014398509481984)