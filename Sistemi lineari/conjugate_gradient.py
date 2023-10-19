"""
Implementation and test for
conjugate gradient method
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg


def conj_grad(A, b, tol=1e-6, dense_output=False):
    """
    Implementation of conjugate gradient
    method for solve A @ x = b; A must be:
    symmetric (i.e. A.T = A)
    positive-definite (i.e. x.T@A@x > 0 for all non-zero x)
    and real.

    Parameters
    ----------
    A : 2darray
        matrix of system.
    b : 1darray
        Ordinate or “dependent variable” values.
    tol : float, optional
        required tollerance default 1e-6
    dense_output : bool, optional
        if True all iteration of th solution, error and number
        of iteration are stored and reurned, default is False

    Return
    ------
    x : 1darray
        solution of system
    err : float
        error of solution

    if dense_output :
    s : 2darray
        all iteration of solution
    e : 1darray
        error of all iteration,
    iter : int
        number of iteration

    """

    N = len(b)
    x = np.zeros(N) #initia guess

    r = b - A @ x   #residuals
    p = r           #descent direction
    r2 = sum(r*r)   #norm^2 residuals

    if dense_output:
        s = []
        e = []
        s.append(x)
        e.append(np.sqrt(r2))

    iter = 0

    while True:

        Ap = A @ p            #computation of
        alpha = r2 /(p @ Ap)  #descent's step

        x = x + alpha * p     #updare position
        r = r - alpha * Ap    #update residuals

        r2_new = sum(r*r)     #norm^2 new residuals
        beta = r2_new/r2      #compute step for p

        r2 = r2_new   #update norm

        if dense_output:
            s.append(x)
            e.append(np.sqrt(r2))

        if np.sqrt(r2_new) < tol : #break condition
            break

        p = r + beta * p   #update p
        iter += 1

    if not dense_output:
        err = np.sqrt(r2_new)
        return x, err
    else:
        return np.array(s), np.array(e), iter


if __name__ == '__main__':

    np.random.seed(69420)

    N = 1000
    P = np.random.normal(size=[N, N])
    A = np.dot(P.T, P) #deve essere simmetrica e semidef >0
    b = np.random.normal(size=[N])

    t1 = time.time()
    sol, err, iter = conj_grad(A, b, 1e-8, dense_output=True)
    x1 = sol[-1]
    t2 = time.time()
    print(f'numero di iterazioni: {iter}')
    print(f'Elapsed time       = {t2 - t1}')

    t1 = time.time()
    x2 = np.linalg.solve(A, b)
    t2 = time.time()
    print(f'Elapsed time numpy = {t2 - t1}')

    t1 = time.time()
    x3, exit_code = cg(A, b)
    t2 = time.time()
    print(f'Elapsed time scipy = {t2 - t1}')

    print('confronto soluzioni')
    print(f"distanza delle due soluzioni(cg-n) = {np.sqrt(np.sum((x1-x2)**2))}")
    print(f"distanza delle due soluzioni(cg-s) = {np.sqrt(np.sum((x1-x3)**2))}")


    plt.figure(1)
    plt.grid()
    plt.plot(abs(err))
    plt.xlabel('iteration')
    plt.ylabel('error')
    #plt.xscale('log')
    plt.yscale('log')
    plt.show()