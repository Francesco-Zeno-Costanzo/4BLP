"""
Code for the solution of Schrodinger's time dependent equation
via split operator method but unlike tunnel_barrier.py using a FFT.
Now the idea is tu use:
exp(1j (T+V) dt) = exp(1j V dt/2) exp(1j T dt) exp(1j V dt/2) + O(dt^3)
and to compute exp(1j T dt) we go in momentum space where T is diagonal
so it easy to compute, so we must use FFT to go from x space to p space
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#=========================================================
# Initial wave function ad potential
#=========================================================

def U(x):
    ''' harmonic potential
    '''
    return 0.5*x**2

def psi_inc(x, x0, a, k):
    ''' Initial wave function
    '''

    A = 1. / np.sqrt( 2 * np.pi * a**2 ) # normalizzation
    K1 = np.exp( - ( x - x0 )**2 / ( 2. * a**2 ) )
    K2 = np.exp( 1j * k * x )
    # let's multiply by five so the animation is prettier
    return A * K1 * K2 * 5

#=========================================================
# Computational parameters
#=========================================================

n  = 1000                    # Number of points
xr = 10                      # Right boundary
xl = -xr                     # Left boundary
L  = xr - xl                 # Size of box
x  = np.linspace(xl, xr, n)  # Grid on x axis
dx = np.diff(x)[0]           # Step size
dt = 1e-3                    # Time step
T  = 10                      # Total time of simulation
ts = int(T/dt)               # Number of step in time

# Initializzation of gaussian wave packet
psi = psi_inc(x, -1.2, 0.5, 0.3)
PSI = []
PSI.append(abs(psi)**2)

#=========================================================
# Build the propagator in x and k space
#=========================================================

# Every possible value of momentum
k   = 2*np.pi*np.fft.fftfreq(n, dx)
# Propagator 
U_r = np.exp(-1j * U(x) * dt/2)  # Half step in space
U_k = np.exp(-1j * k**2/2 * dt)  # Full step in momentum

# Time evolution
for _ in range(ts):
    psi = U_r * psi
    
    psi_k = np.fft.fft(psi)
    psi_k = U_k * psi_k
    
    psi = np.fft.ifft(psi_k)
    psi = U_r * psi
    
    PSI.append(abs(psi)**2)

#=========================================================
# Animation
#=========================================================

fig = plt.figure()
plt.title("Gaussian packet propagation")
plt.plot(x, U(x), label='$V(x)$', color='black')
plt.grid()

plt.ylim(-0.0, np.max(PSI))
plt.xlim(-5, 5)

line, = plt.plot([], [], 'b', label=r"$|\psi(x, t)|^2$")

def animate(i):
    line.set_data(x, PSI[i])
    return line,

plt.legend(loc='best')

anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, ts, 10), 
                               interval=1, blit=True, repeat=True)

#anim.save('ho.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()

