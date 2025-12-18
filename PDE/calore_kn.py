import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def solve_tridiagonal(diag, sup_diag, inf_diag, b):
    '''
    Function to solve a tridiagonal system using Gaussian elimination.

    Parameters
    ----------
    diag : 1darray
        A(i, i) = diag
    sup_diag : 1darray
        A(i, i + 1) = sup_diag
    inf_diag : 1darray
        A(i - 1, i) = inf_diag
    b : array
        RHS of the equation

    Returns
    -------
    x : 1darray
        solution of the sistem
    '''

    N    = len(diag)
    d_ii = np.copy(diag)
    d_up = np.copy(sup_diag)
    d_lo = np.copy(inf_diag)
    b    = np.copy(b)
    x    = np.zeros(N)

    # Forward elimination
    for i in range(1, N):
        if d_ii[i-1] == 0.0:
            raise ValueError("Division by zero, non-invertible matrix")
        factor = d_lo[i-1] / d_ii[i-1]
        d_ii[i] -= factor * d_up[i-1]
        b[i] -= factor * b[i-1]

    # Back substitution
    if d_ii[-1] == 0.0:
        raise ValueError("Division by zero, non-invertible matrix")
    
    x[-1] = b[-1] / d_ii[-1]
    for i in range(N-2, -1, -1):
        b[i] -= d_up[i] * x[i+1]
        if d_ii[i] == 0.0:
            raise ValueError("Division by zero, non-invertible matrix")
        x[i] = b[i] / d_ii[i]
    
    return x

N       = 100              # Numero punti sulle x
T_tesps = 1000             # Numero punti nel tempo
D       = 0.5              # Coefficiente di diffuzione
dx      = 1e-2             # Passo spaziale
dt      = 1e-4             # Passo Temporale
r       = D*dt/(2*dx**2)   # Coefficiente
print(r)

sol_v = np.zeros(N)        # array per le soluzioni
sol_n = np.zeros(N)
# Array per la creazione della matrice
b = np.zeros(N)
a = np.ones(N) * (1 + 2 * r)
c = np.ones(N) * (-r)

# Condizioni Iniali
L = dx * N
t = np.linspace(0, dt*T_tesps, T_tesps)
x = np.linspace(0, L, N)
sol_v = np.exp(-((x - L / 2.0) / 0.1) ** 2)

T = np.zeros((N, T_tesps))
T[:, 0] = sol_v

for time in range(1, T_tesps):
    # Creo il temine noto
    for i in range(1, N-1):
        b[i] = r * sol_v[i+1] + (1 - 2 * r) * sol_v[i] + r * sol_v[i-1]
    b[0] = r * sol_v[1] + (1 - 2 * r) * sol_v[0]
    b[-1] = (1 - 2 * r) * sol_v[-1] + r * sol_v[-2]
    # Risolvo il sistema
    sol_n = solve_tridiagonal(a, c, c, b)
    # Aggiorno e conservo la soluzione
    sol_v = sol_n.copy()
    T[:, time] = sol_v


fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
TT, X = np.meshgrid(t, x)
ax.plot_surface(TT, X, T, linewidth=0,rstride=2, cstride=100)
ax.set_title('Diffusione del calore')
ax.set_xlabel('Tempo')
ax.set_ylabel('Lunghezza')
ax.set_zlabel('Temperatura')

fig = plt.figure(2)
plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(T), np.max(T))

line, = plt.plot([], [], 'b')
def animate(i):
    line.set_data(x, T[:,i])
    return line,


anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, T_tesps, 10), interval=10, blit=True, repeat=True)

plt.grid()
plt.title('Diffusione del calore')
plt.xlabel('Distanza')
plt.ylabel('Temperatura')

#anim.save('calore.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# Analytical solution
N = 2000
sol = 0*X
for n in range(1, N):
    y = np.sin(n * np.pi * x/L) * np.exp(-((x - L/2)/0.1)**2)
    A_n = integrate.simpson(y, x=x)
    sol += 2/L * A_n * np.sin( n * np.pi * X / L) * np.exp(-D * (n*np.pi/L)**2 * TT)

# avoid strange problems
sol[:, 0] = np.exp(-((x - L / 2.0) / 0.1) ** 2)

fig = plt.figure(5)
error0 = abs((T  - sol)**1)

levels0 = np.linspace(np.min(error0), np.max(error0), 40)

c=plt.contourf(X, TT, error0, levels=levels0, cmap='plasma')
plt.colorbar(c)

plt.show()