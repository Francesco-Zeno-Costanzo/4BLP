"""
newton method for nonlinear system of equations
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root

#=============================================================
# Newron method implementation
#=============================================================

def newton(f, start, tol, args=(), dense_output=False, max_it=1000):
    '''
    Generalizzation of newton method for n equations
    
    Parameters
    ----------
    f : callable
        A vector function to find a root of.
    start : ndarray
        Initial guess. The method is very sensitive to this
        must be chosen carefully
    tol :float, optional, default 1e-8
        required tollerance
    args : tuple, optional
        Extra arguments passed to f
    dense_output : bool, optional
        true for full and number of iteration
    max_it : int, optional, default 1000
        after max_it iteration the code stop raising an exception
    
    Return
    ------
    x0 : 1darray
        solution of the system
        if dense_outupt=True all iteration are returned
        in a matrix called X and also the number of iteration
    '''
    
    # initial guess
    x0 = start
    f0 = f(x0, *args)
    # for the computation of jacobian
    nd = len(x0)
    df = np.zeros((nd, nd))
    h  = 1e-8
    s  = np.zeros(nd)
    #for full output
    X = []
    if dense_output : X.append(x0)
    # count
    n_iter = 0
    
    while True:
        
        # compute jacobian using symmetric derivative
        for i in range(nd):       # loop over functions
            for j in range(nd):   # loop over variables
                s[j] = 1
                xr, xl = x0 + h*s, x0 - h*s
                df[i, j] = (f(xr, *args) - f(xl, *args) )[i]/(2*h)
                s[:] = 0
        
        # update solution
        delta = np.linalg.solve(df, f0)
        x0 = x0 - delta
        f0 = f(x0, *args)
        
        if dense_output:
            n_iter += 1        
            X.append(x0)
        
        # stop condition
        if all(abs(f0) < tol):
            break
        # check iterations
        if n_iter > max_it :
            err_msg = 'too many iteration, failure to converge, change initial guess'
            raise Exception(err_msg) 
    
    if dense_output:
        return np.array(X), n_iter       
    else:
        return x0

#=============================================================
# System to solve
#=============================================================        

def system(V):
    x1, x2 = V
    
    r1 = x1**2 + x2**2 - 1#x3
    r2 = x2 - x1**2 + x1/2 #+ x3/5
    #r3 = x3**2 + 5*x2 - 7
    
    R = np.array([r1, r2])#, r3])
    return R

#=============================================================
# Solution and copare with scipy
#=============================================================    

init = np.array([0.2, 0.0])
tol = 1e-12
sol, n_iter = newton(system, init, tol=tol, dense_output=True)
xs, ys = sol.T
print("Solution with newton: ", *sol[-1], "in", n_iter, "iterations")

sol = root(system, init, method='hybr', tol=tol)
print("Solution with root:   ", *sol.x)

sol = fsolve(system , init, xtol=tol)
print("Solution with fsolve: ", *sol)

#=============================================================
# Plot
#=============================================================    

t = np.linspace(0, 2*np.pi, 1000)
z = np.linspace(-0.8, 1.5, 1000)
plt.figure(1)
plt.grid()
plt.plot(np.cos(t), np.sin(t), 'r', label='first equation')
plt.plot(z, z**2- z/2, 'b', label='second equation')
plt.plot(xs, ys, 'k', label='evolution of solution')
plt.legend(loc='best')
plt.show()



