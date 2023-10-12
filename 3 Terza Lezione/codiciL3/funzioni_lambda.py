f = lambda x : 3*x
print(f(2))

h = lambda x, y, z : x*y + z
print(h(2, 3, 4))

#============================================
#============================================

def g(x):
    '''restituisce potenza x-esima di y
    '''
    return lambda y : y**x

G = g(3) # potenza cubica
print(G(4))

#============================================
#============================================

def m(f, y, x):
    '''passo una funzione ad una funzione
    '''
    return f(y) - x

print(m(lambda x : x**2, 5, 1))