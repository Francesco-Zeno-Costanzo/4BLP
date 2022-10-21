import numpy as np
import matplotlib.pyplot as plt

# geometric Brownian motion
# Heun method

def f(z):
    """
    funzione che moltiplica il dt
    """
    mu = 1
    return mu*z

def g(z):
    """
    funzione che moltimplica il processo di wiener
    """
    sigma = 0.5
    return sigma*z

def dg():
    """
    derivata di g
    """
    sigma = 0.5
    return sigma

def dW(delta_t):
    """
    processo di wiener trattato come variabile gaussiana
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))


#parametri simulazioni
N = 10000
tf = 4
dt = tf/N
#faccio 5 simulazioni diverse
for _ in range(5):
    #array dove conservare la soluzione, ogni volta inizializzati
    ts = np.zeros(N + 1)
    ys = np.zeros(N + 1)

    ys[0] = 1#condizioni iniziali

    for i in range(N):
        ts[i+1] = ts[i] + dt
        y0 = ys[i] + f(ys[i])*dt + g(ys[i])*dW(dt) + 0.5*g(ys[i])*dg()*(dW(dt)**2)
        y1 = ys[i] + f(y0)*dt    + g(y0)*dW(dt)    + 0.5*g(y0)*dg()*(dW(dt)**2)
        ys[i+1] = 0.5*(y0 + y1)

    plt.plot(ts, ys)

plt.figure(1)
plt.title('moto geometrico Browniano')
plt.xlabel("time")
plt.grid()

plt.show()