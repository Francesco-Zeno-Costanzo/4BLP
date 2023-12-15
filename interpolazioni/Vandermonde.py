import numpy as np
import matplotlib.pyplot as plt

N = 10
x = np.linspace(0, 1, N)
y = np.sin(2*np.pi*x)

#Matrice di Vandermonde
A = np.zeros((N, N))
A[:,0] = 1
for i in range(1, N):
    A[:,i] = x**i

#risolvo il sistema, la soluzione sono i coefficenti del polinomio
s = np.linalg.solve(A, y)


def f(s, zz):
    '''
    funzione per fare il grafico
    '''
    n = len(zz)
    y = np.zeros(n)
    for i , z in enumerate(zz):
        y[i] = sum([s[j]*z**j for j in range(len(s))])
    return y

z = np.linspace(0, 1, 100)

plt.figure(1)
plt.title('Interpolazione polinomiale')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(z, f(s, z), 'b', label='interpolazione')
plt.plot(x, y, marker='.', linestyle='', c='k', label='dati')
plt.legend(loc='best')
plt.grid()
plt.show()