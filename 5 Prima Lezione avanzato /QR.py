import time
import numpy as np
import matplotlib.pyplot as plt

#=====================================================================
# Function to compute QR decomposition
#=====================================================================

def QR_decomp(A):
    '''
    Comupute QR decomposition of a matrix A
    
    Parameters
    ----------
    A : 2darray
        N x N matrix
    
    Returns
    -------
    Q, R : 2darray
        A = Q @ R
    '''
    
    # we give different names because in principle the QR
    # decomposition also applies to non-square matrices
    n, m = A.shape

    Q = np.zeros((n, n)) # Initialize matrix Q
    R = np.zeros((n, m)) # Initialize matrix R
    u = np.zeros((n, n)) # Initialize matrix u: for Gram-Schmidt

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.sqrt(sum(u[:, 0]**2))

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j]

        Q[:, i] = u[:, i] / np.sqrt(sum(u[:, i]**2))

    
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]
    
    # uncomment for scipy comparison      
    #D = np.diag(np.sign(np.diag(Q)))
    #Q[:, :] = Q @ D
    #R[:, :] = D @ R

    return Q, R

#=====================================================================
# Function to solve eigensystem via QR iterartion
#=====================================================================

def QR_eig(A, maxiter=100):
    '''
    Find the eigenvalues of A using QR iteration.
    
    Parameters
    ----------
    A : 2darray
        N x N matrix
    maxiter : int, optional, default 100
        number of iterations to do
    
    Returns
    -------
    eigval : 1darray
        array of eigenvalues
    eigvec : 2darray
        N x N matrix, the column eigvec[:, i] is the
        ormalized eigenvector corresponding to the
        eigenvalue eigval[i]
    '''
    A_new, A_old = [np.copy(A)]*2
   
    eigvec = np.eye(A.shape[0])
      
    for i in range(maxiter):
        
        A_old[:, :] = A_new
        Q, R = QR_decomp(A_old)

        A_new[:, :] = R @ Q
        eigvec = eigvec @ Q       

    eigval = np.diag(A_new)

    return eigval, eigvec


def test(n, qriter):
    
    P = np.random.normal(size=[n, n])
    H = np.dot(P.T, P)  #for symmetric matrix
    
    start = time.time()
    eigval1, eigvec1 = QR_eig(H, qriter)
    end = time.time() - start
    print(f'Elapsed time       = {end:.5f}')
    
    start = time.time()
    eigval2, eigvec2 = np.linalg.eigh(H)
    end = time.time() - start
    print(f'Elapsed time numpy = {end:.5f}') 
    
    # Sort
    eigval1s = np.sort(eigval1)
    eigval2s = np.sort(eigval2)
    eigvec1s = eigvec1[:,eigval1.argsort()[::-1]]
    eigvec2s = eigvec2[:,eigval2.argsort()[::-1]]
    
    diff_l1 = abs(eigval1s-eigval2s)
    
    diff_n1 = np.zeros(n)
    
    for i in range(n):

        ps = np.dot(eigvec1s[:,i], eigvec2s[:,i])
        if ps < 0:      #if v is an eigenvector
            diff_n1[i] = np.sqrt(sum((eigvec1s[:,i]+eigvec2s[:,i])**2))
        else :          #also -v is an eigenvectors
            diff_n1[i] = np.sqrt(sum((eigvec1s[:,i]-eigvec2s[:,i])**2))
    
    plt.figure(0)

    plt.subplot(211)
    plt.title('differce between eigenvalues')
    plt.ylabel(r'$abs( \lambda - \lambda_{numpy})$')
    plt.yscale('log')
    plt.grid()
    plt.plot(diff_l1, marker='.', linestyle='', color='blue')

    plt.subplot(212)
    plt.title('differce between eigenvectors')
    plt.ylabel(r'$|\vec{v} - \vec{v}_{numpy}|$')
    plt.yscale('log')
    plt.grid()
    plt.plot(diff_n1, marker='.', linestyle='', color='blue')
    

if __name__ == "__main__":

    np.random.seed(69420)
    
    test(100, 400)
    plt.show()
