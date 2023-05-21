import time
import numpy as np

def gauss_sidel(A, b, tol):
    """
    Implementation of gauss_sidel method for solve A @ x = b;
    A must be:
    1) symmetric (i.e. A.T = A)
    2) positive-definite (i.e. x.T@A@x > 0 for all non-zero x)
    3) real.

    Parameters
    ----------
    A : 2darray
        matrix of system.
    b : 1darray
        Ordinate or “dependent variable” values.
    tol : float, optional
        required tollerance default 1e-6

    Return
    ------
    x : 1darray
        solution of system
    """
    x = np.zeros(len(b))
    iter = 0
    while True:

        x_new = np.zeros(len(x))

        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        res = np.sqrt(np.sum((A @ x_new - b)**2))
        if res < tol:
            break

        x = x_new
        iter += 1

    return x, iter

if __name__ == "__main__":
    np.random.seed(69420)
    N = 13
    A = np.random.normal(size=[N, N])
    A = A.T @ A #per garantire la convergenza del metodo
    b = np.random.normal(size=[N])

    start = time.time()
    x1, iter = gauss_sidel(A, b, 1e-8)
    print(f"number of iteration = {iter}")
    print(f"Elapsed time = {time.time()-start}")

    start = time.time()
    x2 = np.linalg.solve(A, b)
    print(f"Elapsed time = {time.time()-start}")


    d = np.sqrt(np.sum((x1 - x2)**2))
    print(f'difference with numpy = {d}')