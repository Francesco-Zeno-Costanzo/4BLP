import numpy as np
import matplotlib.pyplot as plt

# Ornstein–Uhlenbeck
# Euler–Maruyama method

def f(z):
    """
    funzione che moltiplica il dt
    """
    theta = 0.7
    mu = 2.5
    return theta * (mu - z)

def g():
    """
    funzione che moltimplica il processo di wiener
    """
    sigma = 0.6
    return sigma

def dW(delta_t):
    """
    processo di wiener trattato come variabile gaussiana
    """
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

#parametri simulazione
N = 10000
tf = 15
dt = tf/N

ts = np.zeros(N + 1)
ys = np.zeros(N + 1)
xs = np.zeros(N + 1)

ys[0], xs[0] = 0, 0 #condizioni inizali

for i in range(N):
    ys[i+1] = ys[i] + f(ys[i]) * dt + g() * dW(dt)
    xs[i+1] = xs[i] + f(xs[i]) * dt + g() * dW(dt)
    ts[i+1] = ts[i] + dt

plt.figure(1)
plt.plot(xs, ys)
plt.title('Ornstein–Uhlenbeck')
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

plt.figure(2)
plt.plot(ts, xs, label='x(t)')
plt.plot(ts, ys, label='y(t)')
plt.title('Ornstein–Uhlenbeck')
plt.xlabel("t")
plt.legend()
plt.grid()

plt.show()