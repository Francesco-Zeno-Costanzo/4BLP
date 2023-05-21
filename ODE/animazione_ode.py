import numpy as np
import scipy.integrate
from matplotlib import animation
import  matplotlib.pyplot as plt

#parametri
N = 10000       #numero di punti
l = 1           #lunghezza pendolo
g = 9.81        #accellerazione di gravita'
o0 = g/l        #frequenza piccole oscillazioni
v0 = 0          #condizioni iniziali velocita'
x0 = np.pi/1.1	 #condizioni iniziali posizione
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

#passaggio in cartesiane
theta = sol[:,0]
x = l*np.sin(theta)
y = -l*np.cos(theta)

#grafico e bellurie
fig = plt.figure(1, figsize=(10, 6))
plt.suptitle('Pendolo semplice')
ax = fig.add_subplot(121)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect('equal', adjustable='box')

#coordinate del perno e della pallina
xf, yf = [0,x[0]],[0,y[0]]

line1, = plt.plot(xf, yf, linestyle='-', marker='o',color='k')

plt.grid()

def animate(i):
    """
    funzione che a ogni i aggiorna le corrdinate della pallina
    """
    xf[1] = x[i]
    yf[1] = y[i]
    line1.set_data(xf, yf)
    time_text.set_text(time_template % (i*t[1]))

    return line1, time_text

#funzione che fa l'animazione vera e propria
anim = animation.FuncAnimation(fig, animate, frames=range(0, len(t), 5), interval=1, blit=True, repeat=True)

plt.subplot(122)
plt.ylabel(r'$\theta$(t) [rad]')
plt.xlabel('t [s]')
plt.plot(t, theta)
plt.grid()
plt.show()