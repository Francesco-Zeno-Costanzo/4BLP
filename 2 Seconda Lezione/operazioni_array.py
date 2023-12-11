import math
import numpy as np

v = np.array([4, 5, 6])
w = np.array([1.2, 3.4, 5.8])

#classiche operazioni
somma = v + w
sottr = v - w
molt = v * w
div = v / w

print(v, w)
print()
print(somma, sottr, molt, div)
print()
#altri esempi
print(v**2)
print(np.log10(w))

"""
come dicevamo prima qui' otterremmo errore poiche'
math lavora solo con numeri o, volendo,
array unidimensionali lunghi uno
"""
print(math.log10(w))
