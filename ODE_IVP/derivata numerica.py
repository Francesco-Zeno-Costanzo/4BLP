import numpy as np
import matplotlib.pyplot as plt

def f(x):
    '''
    funzione di cui calcolare la derivata
    '''
    return np.exp(x)

def df(f, x, h):
    """
    derivata di f
    """
    dy = (f(x+h) - f(x))/h
    return dy

#array del passo di discretizzazione
h = np.logspace(-15, -1, 1000)

plt.figure(1)
plt.title('errore derivata al variare del passo', fontsize=15)
plt.ylabel('erorre derivata', fontsize=15)
plt.xlabel("grandezza dell'incremento", fontsize=15)

plt.plot(h, abs(df(f, 0, h)-f(0)))

plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()