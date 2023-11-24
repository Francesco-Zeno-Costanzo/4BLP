import time
import numpy as np
import matplotlib.pyplot as plt

def inverse(M, check=True):
    '''
    Compute inverse of M by Gauss elimination

    Parameters
    ----------
    M : 2d array
        matrix to invert

    Return
    ------
    M : 2d array
        inverse matrix
    '''

    n, m = M.shape
    if check :
        if m != n:
            msg_err = " A must be square "
            raise ValueError(msg_err)

        if abs(np.linalg.det(M)) < 1e-15: # determinant O(N^3)
            msg_err = " det(A) = 0 non-invertible "
            raise ValueError(msg_err)
    
    I = np.identity(n=n)        # Indentity
    A = np.zeros((n, 2*n))
    A[:, :n], A[:, n:] = M, I   # concatenating M and I
    
    for i in range(0, n):   # loop over matrix rows

        j = 1               # initialize row-swap iterator
        pivot = A[i][i]     # select pivot value

        while pivot == 0 and i + j < n:   # next non-zero leading coefficient

            A[[i, i + j]] = A[[i + j, i]] # row swap
            j += 1                        # incrememnt row-swap iterator
            pivot = A[i][i]               # get new pivot

        if pivot == 0:      # if pivot is zero, remaining rows are all zeros
            return A[:, n:] # return inverse matrix

        row = A[i]          # extract row
        A[i] = row / pivot  # get 1 along the diagonal
        # iterate over all rows except pivot
        for j in [k for k in range(0, n) if k != i]:
            # subtract current row from remaining rows
            A[j] = A[j] - A[i] * A[j][i]
            
    return A[:, n:] # return inverse matrix


if __name__ == '__main__':
    np.random.seed(69420)
    N = 100
    A = np.random.rand(N, N)
    
    start = time.time()
    A1 = inverse(A, False)
    print(f" Elapsed time = {time.time() - start}")

    start = time.time()
    A2 = np.linalg.inv(A)
    print(f" Elapsed time = {time.time() - start}")

    plt.figure(1)
    plt.plot(abs((A1-A2).ravel()), marker='.', linestyle='')
    plt.yscale('log')
    plt.grid()
    plt.show()
