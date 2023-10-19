import numpy as np
from scipy.sparse import diags
import  matplotlib.pyplot as plt

#=================================================================================
# Function for the solution of boundary value problem via relaxation
#=================================================================================

def relax(f, y0, y1, x, init, args=(), tol=1e-8, max_iter=100, dense_output=False):
    '''
    Implementation of relaxation method for ODE 2pt-BVP.
    The equation must be in the form: y''(x) = f(x, y, y')

    Parameters
    ----------
    f : callable
        A vector function of differential equation like: y'' = f
    y0, y1 : float
        required value of solution at boundary
    x : 1darray
        array of position, or time, independent variable
    init : 1darray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to f
    tol :float, optional, default 1e-8
        required tollerance
    max_iter : int, optional, default 100
        after max_it iteration the code stop raising an exception
    dense_output : bool, optional, default False
        true for full and number of iteration

    Return
    ------
    yo : 1darray
        solution of differential equation
        if dense_outupt=True all iteration are returned
        in a matrix called Y and also the number of iteration
    '''
    # parameter of discretizzation
    N = len(x)
    h = np.diff(x)[0]
    # second derivative matrix
    d2 = diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).toarray()
    d2 = d2/h**2
    # bound values required
    yb = [y0/h**2] + [0]*int(N-2) + [y1/h**2]
    yb = np.array(yb)
    # init guess from imput
    yo = init
    # interation count
    it = 0
    #for full output
    Y = []
    if dense_output : Y.append(yo)

    while True:
        # for jacobian computation
        df = np.zeros(d2.shape)
        s  = np.zeros(N)

        for i in range(N):
            s[i] = 1
            yr, yl = yo + h*s, yo - h*s
            df[i, :] = (f(x, yr, h, *args) - f(x, yl, h, *args) )/(2*h)
            s[:] = 0

        yn = yo - np.linalg.solve(d2-df, d2@yo - f(x, yo, h, g, o02) + yb)
        # residual
        R = np.sqrt(np.sum((yn-yo)**2))
        if R < tol:
            yo = yn
            break

        if it > max_iter:
            raise Exception("to many iteration")

        #update
        yo = yn
        it = it + 1
        if dense_output : Y.append(yo)

    if dense_output:
        return Y, it
    else:
        return yo

#=================================================================================
# RHS of differential equations y'' = f
#=================================================================================

def f(t, y, h, g, o02):
    '''
    RHS of differential equations y'' = f
    f can be a function non only of y but also y' so
    we use a second order approximation to compute y'

    Parameter
    ---------
    t : 1darray
        independent variable
    y : 1darray
        solution or guess of solution
    h : float
        step's size for derivative computation
    g, o02 : float
        parameter of our differential equation

    Return
    ------
    y_ddot : float
        RHS of equation computed on a grid
    '''
    y_dot   = (y[2:] - y[:-2])/(2*h)              # second order derivative
    y_dot_0 = (- 3*y[0]  + 4*y[1]  - y[2] )/(2*h) # second order derivative on left  bound
    y_dot_n = (  3*y[-1] - 4*y[-2] + y[-3])/(2*h) # second order derivative on right bound
    # join everything together to get the second order derivative
    y_dot = np.insert(y_dot, 0,          y_dot_0)
    y_dot = np.insert(y_dot, len(y_dot), y_dot_n)

    # equation to solve
    y_ddot = -g*y_dot - o02*y

    return y_ddot

#=================================================================================
# Main code and plot
#=================================================================================

g   = 0.3    # damping factor
o02 = 1      # proper frequency squared
xi  = 0      # left  end of the interval
xf  = 5      # right end of the interval
N   = 1000   # number of points
y0  = 1      # boundary condition on xi
y1  = 0.1862 # boundary condition on xf

x = np.linspace(xi, xf, N)
y = (x - xi) * (y1 - y0)/(xf - xi) + y0 # linear guess

Y, n = relax(f, y0, y1, x, y,args=(g, o02), dense_output=True)
print(f"{n} iterations required")

for y in Y:
    plt.plot(x, y)

plt.title("Solution via relaxation method", fontsize=15)
plt.xlabel("x", fontsize=15)
plt.ylabel("y", fontsize=15)
plt.grid()
plt.show()
