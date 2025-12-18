"""
Code for the solution of Laplace equation via SOR
"""
import time
import numpy as np
import matplotlib.pyplot as plt

def solve_lap_sor(N, bound, rho, w, tau):
    """
    Function that use SOR method to solve laplace equation

    Parameters
    ----------
    N : int
        size of the grid
    bound : 2darray
        boundary conditions
    rho : 2darray
        source of field
    w : float,
        overrelaxation parameter
    tau : float
        required convergence
    
    Return
    ------
    phi : 2darray
        solution of the equations
    """
    # Unsefuel quantities
    conv       = 10.0
    mean_U0    = 0
    iter_count = 0
    
    # Matrix of the potential
    phi = np.zeros((N+1, N+1))
    
    # Boundary conditions
    phi[:, 0] = bound[0]  # ovest
    phi[:, N] = bound[1]  # est
    phi[0, :] = bound[2]  # sud
    phi[N, :] = bound[3]  # nord

    mean_U0 = np.mean(phi)
    
    while conv > tau:
        for i in range(1, N):
            for j in range(1, N):
                force = phi[i, j+1] + phi[i, j-1] + phi[i+1, j] + phi[i-1, j]
                force += rho[i, j]
                phi[i, j] = w * 0.25 * force + (1 - w) * phi[i, j] #SOR

        mean_U  = np.mean(phi)
        conv    = abs(mean_U - mean_U0)
        mean_U0 = mean_U

        iter_count += 1

    return phi, iter_count


if __name__ == "__main__":
    N   = 100
    w   = 1.99
    tau = 1e-8
    dx  = 1/N

    x = np.linspace(-1, 1, N+1)
    # Boundary conditions
    bound = np.zeros((4, N+1))
    bound[0, :] = 0.5*(x**2 -x)#0   # ovest
    bound[1, :] = 1/(1/np.e - np.e)*(np.exp(x)-np.e) #0  # est
    bound[2, :] = x**4#-2  # sud
    bound[3, :] = np.cos(np.pi*x/2)#2   # nord
    
    # Source
    rho = np.zeros((N+1, N+1))
    for i in range(-1, 2):
        for j in range(-1, 2):
            rho[7*N//11+i, N//2+j] = -10000
            rho[4*N//11+i, N//2+j] = 10000
    rho = 0*rho*dx**2

    # Solution
    start = time.time()
    phi, ic = solve_lap_sor(N, bound, rho, w, tau)
    print(ic)
    end = time.time() - start

    print(f"Elapsed time: {end:.2f} s")

    # Plot
    #x = np.linspace(0, 1, N+1)
    gridx, gridy = np.meshgrid(x, x)

    plt.figure(0)
    levels = np.linspace(np.min(phi), np.max(phi), 40)
    c=plt.contourf(gridx, gridy, phi , levels=levels, cmap='plasma')
    plt.colorbar(c, label='Potential')
    plt.title('Laplace Equation Solution with SOR Method')
    plt.xlabel('x')
    plt.ylabel('y')

    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.plot_surface(gridx, gridy, phi, cmap='plasma')
    ax.set_title('Solution of laplace equation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('potential')

    plt.show()
