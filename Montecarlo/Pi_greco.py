import time
import numpy as np
import matplotlib.pyplot as plt

N = int(5e4)
start_time=time.time()

x = np.linspace(0,1, 10000)

def f(x):
    """
    semi circonferenza superiore
    """
    return np.sqrt(1-x**2)

c = 0

for i in range(1,N):
    #genero due variabili casuali uniformi fra 0 e 1
    a = np.random.rand()
    b = np.random.rand()
    r = a**2 + b**2
    #se vero aggiorno c di 1
    if r < 1:
        plt.errorbar(a, b, fmt='.', markersize=1, color='blue')
        c += 1
    else:
        plt.errorbar(a, b, fmt='.', markersize=1,  color='green')

#moltiplico per quattro essendo su un solo quadrante
Pi = 4*c/N
#propagazione errore, viene dalla binomiale
dPi = np.sqrt(c/N * (1-c/N))/np.sqrt(N)
print('%f +- %f' %(Pi, dPi))
print(np.pi)
print(abs((Pi-np.pi)/np.pi))

plt.figure(1)
plt.title('Pi $\simeq$ %f $\pm$ %f ; N=%.0e' %(Pi, dPi, N), fontsize=20)
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(x, f(x),color='blue', lw=1)
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))