import numpy as np
import scipy.integrate
import  matplotlib.pyplot as plt

#parametri
N = 100000      #numero di punti
l = 1           #lunghezza pendolo
g = 9.81        #accellerazione di gravita'
o0 = g/l        #frequenza piccole oscillazioni
v0 = 0          #condizioni iniziali velocita'
x0 = np.pi/1.1	#condizioni iniziali posizione
tf = 15         #fin dove integrare

#odeint
def ODE_1(y, t):
    """
    equzione da risolvere per odeint
    """
    theta, omega = y
    dydt = [omega,  - o0*np.sin(theta)]
    return dydt


y0 = [x0, v0] #x(0), x'(0)
t = np.linspace(0, tf, N+1)
sol = scipy.integrate.odeint(ODE_1, y0, t)

x_scipy = sol[:,0]

#metodo di eulero
def ODE_2(x, v):
    """
    equzione da risolvere per eulero
    """
    x_dot = v
    v_dot = -o0*np.sin(x)
    return x_dot, v_dot

def eulero(N, tf, x0, v0):
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
    v = np.zeros(N+1)
    x[0], v[0] = x0, v0

    for i in range(N):
        dx, dv = ODE_2(x[i], v[i])
        x[i+1] = x[i] + dt*dx
        v[i+1] = v[i] + dt*dv

    return x, v

x_eulero, _ = eulero(N, tf, x0, v0)


plt.figure(1)

plt.title('Pendolo semplice')
plt.xlabel('t')
plt.ylabel('x')
plt.plot(t, x_scipy, label='scipy')
plt.plot(t, x_eulero, label='elulero')
plt.legend(loc='best')
plt.grid()

plt.show()