import numpy as np
import scipy.integrate
import  matplotlib.pyplot as plt

#parametri
x0 = 1     #condizione inizale
tf = 2     #fino a dove integrare
N = 10000  #numero di punti

#odeint
def ODE_1(y, t):
    """
    equzione da risolvere per odeint
    """
    x = y
    dydt = x
    return dydt


y0 = [x0] #x(0)
t = np.linspace(0, tf, N+1)
sol = scipy.integrate.odeint(ODE_1, y0, t)

x_scipy = sol[:,0]

#metodo di eulero
def ODE_2(x):
    """
    equzione da risolvere per eulero
    """
    x_dot = x
    return x_dot

def eulero(N, tf, x0):
    """
    si usa che dx/dt = (x[i+1]-x[i])/dt
    che e' praticamente la definizione di rapporto incrementale
    discretizzata la derivata sappiamo a cosa eguagliarla
    perche dx/dt = g(x(t)) nella fattispecie g(x) = x
    quindi discretizzando tutto:
    (x[i+1]-x[i])/dt = x[i]
    da cui si isola x[i+1]
    """
    dt = tf/N #passo di integrazione
    x = np.zeros(N+1)
    x[0] = x0

    for i in range(N):
        x[i+1] = x[i] + dt*ODE_2(x[i])

    return x

x_eulero = eulero(N, tf, x0)

plt.figure(1)

ax1 = plt.subplot(211)
ax1.set_title('Risoluzione numerica')
ax1.set_xlabel('t')
ax1.set_ylabel('x')
ax1.plot(t, x_scipy, label='scipy')
ax1.plot(t, x_eulero, label='elulero')
ax1.legend(loc='best')
ax1.grid()

ax2 = plt.subplot(223)
ax2.set_title('Differenza tra metodo di eulero e soluzione esatta')
ax2.set_xlabel('t')
ax2.set_ylabel('x')
ax2.plot(t, x_eulero-np.exp(t))
ax2.grid()


ax3 = plt.subplot(224)
ax3.set_title('Differenza tra odeint e soluzione esatta')
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.plot(t, x_scipy-np.exp(t))
ax3.grid()

plt.show()