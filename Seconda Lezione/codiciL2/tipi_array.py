import numpy as np

array1 = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

tipoarray1 = array1.dtype
print(tipoarray1)

a = np.array([0, 1, 2])
#abbiamo scritto solo numeri interi => array di interi

b = np.array([0., 1., 2.])
#abbiamo scritto solo numeri con la virgola => array di numeri float

"""
#nota: anche se si dice "numero con la virgola",
vanno scritti sempre col punto!
La virgola separa gli argomenti
"""

c = np.array([0, 3.14, 'giallo'])
#quest'array e' misto. Ci sono sia numeri interi che float che stringhe


#ora invece il tipo viene definito in maniera esplicita:
d = np.array([0., 1., 2.], 'int')
e = np.array([0, 1, 2], 'float')

print(a, a.dtype)
print(b, b.dtype)
print(c, c.dtype)
print(d, d.dtype)
print(e, e.dtype)