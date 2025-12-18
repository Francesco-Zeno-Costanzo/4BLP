"""
code to solve burger equation using fourier trasform in space
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

#=============================================================
# Parameters
#=============================================================

L = 1
T = 0.9

dx = 0.001
dt = 0.001
nu = 0.005

Nx = int(L/dx)
Nt = int(T/dt)

t = np.linspace(0, T, Nt)
x = np.linspace(0, L, Nx)

# wave number
k = 2*np.pi*np.fft.fftfreq(Nx, dx)
# Initial conditions
u0 = np.sin(2*np.pi*x)
u0 = np.concatenate((np.ones(Nx//3), -3*x[Nx//3:2*Nx//3] + 2, 0*x[2*Nx//3:]))

#=============================================================
# Solutions
#=============================================================

def eq(u, t, k, nu):
    '''
    equation to be solved in spatial transform
    '''
    u_hat = np.fft.fft(u)
    du_hat = 1j*k*u_hat    # first  derivative
    ddu_hat = -k**2*u_hat  # second derivative

    #antitrasform
    du = np.fft.ifft(du_hat)
    ddu = np.fft.ifft(ddu_hat)

    #pde in time and space -> ode in time
    u_t = -u*du + nu*ddu

    return u_t.real


sol = odeint(eq, u0, t, args=(k, nu,)).T


#=============================================================
# Animations
#=============================================================

fig = plt.figure(2)
plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(sol)-0.1, np.max(sol)+0.1)

line, = plt.plot([], [], 'b')
def animate(i):
    line.set_data(x, sol[:,i])
    return line,

anim = animation.FuncAnimation(fig, animate, frames=Nt, interval=5, blit=True, repeat=True)

plt.grid()
plt.title('burger equation')
plt.xlabel('Distanza')
plt.ylabel('ampiezza')

#anim.save('buger.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

X, Y = np.meshgrid(x, t)
fig = plt.figure(0)
plt.contourf(X, Y, sol.T)
#ax.plot_surface(X, Y, sol.T, cmap='plasma',linewidth=0,rstride=2, cstride=100)
#ax.set_title('Heat diffussion')
#ax.set_xlabel('Time')
#ax.set_ylabel('Distance')
#ax.set_zlabel('Temperature')
plt.show()
