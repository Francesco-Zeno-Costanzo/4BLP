import time
import numpy as np
import scipy.special as ssp
from scipy.sparse import diags
import matplotlib.pyplot as plt

from QR import QR_eig
from eigvec import eig


def lanczos(A, n):
    '''
    Lanczos iteration for a matrix A
    
    Parameter
    ---------
    A : 2darray
        Hermitian N x N matrix
    n : int
        dimension of Krylov subspace
        e.g. dimension of H
    
    Return
    ------
    Q, H : 2darray
        A = Q.T H Q
    '''
    m     = A.shape[0]
    Q     = np.zeros((m, n+1))
    alpha = np.zeros(n)
    beta  = np.zeros(n)
    b     = np.random.randn(m)
    
    Q[:,0] = b/np.sqrt(sum(b**2))
    
    for i in range(n):
        v        = np.dot(A, Q[:,i])
        alpha[i] = np.dot(Q[:,i], v)
        
        if i == 0:
            v = v - alpha[i] * Q[:, i]
        else :
            v = v - beta[i-1] * Q[:, i-1] - alpha[i] * Q[:, i]
        
        beta[i]  = np.sqrt(sum(v**2))
        Q[:,i+1] = v / beta[i]
    
    H = Q.T @ A @ Q
    
    return Q, H

 
if __name__ == "__main__":

    np.random.seed(69420)
    
#===============================================================================
# Computational parameter
#===============================================================================
         
    k  = 10                    # how many levels compute for power method
    n  = 1000                  # size of matrix
    xr = 10                    # bound
    xl = -10                   # bound
    L  = xr - xl               # dimension of box
    h  = (xr - xl)/(n)         # step size
    tt = np.linspace(0, n, n)  # array form 0 to n
    xp = xl + h*tt             # array of position
    
#===============================================================================
# Hamiltonian
#===============================================================================

    P = diags([1, -2, 1], [-1, 0, 1], shape=(n, n)).toarray()
    V = diags(0.5*xp**2, 0, shape=(n, n)).toarray()
    A = -(1/(2*h**2))*P + V
    Q, H = lanczos(A, 300)

#===============================================================================
# Computation
#===============================================================================
    
    start = time.time()
    #eigval, eigvec = QR_eig(H)
    eigval, eigvec, _ = eig(H, k, tol=1e-5)
    
    eigvec = Q @ eigvec
    
    eigvec = eigvec[:,eigval.argsort()]
    eigval = np.sort(eigval)  
    
    end = time.time() - start
    print(f'Elapsed time       = {end:.5f}\n')
    
    print("Theoretical     Computed          error")
    print("-----------------------------------------")
    for i in range(k):
        print(f'{i + 0.5} \t \t {eigval[i]:.5f} \t {eigval[i]-(i+0.5):.2e}')
        #print(f'{i + 0.5} & {eigval1[i]:.5f} & {eigval1[i]-(i+0.5):.2e} & {eigval2[i]:.5f} & {eigval2[i]-(i+0.5):.2e}'+'\\'+'\\')

    psi = eigvec/np.sqrt(h)
    
#===============================================================================
# Plot
#===============================================================================
    
    def G(x, m):
        return (1/(np.pi)**(1/4))*(1/np.sqrt((2**m)*ssp.gamma(m+1)))*ssp.eval_hermite(m, x)*np.exp(-(x**2)/2)

    plt.figure(1)
    plt.title("Oscillatore armonico", fontsize=15)
    plt.xlabel('x', fontsize=15)
    plt.ylabel(r'$\psi(x)$', fontsize=15)
    plt.grid()
    plt.ylim(0, 3)
    plt.xlim(-5, 5)

    plt.plot(xp, 0.5*xp**2, color='black', label='V(x)= $ x^2/2 $')
    c = ['b', 'r', 'g']
    
    for L in range(3):

        plt.errorbar(xp, abs(psi[:,L])**2 + eigval[L], color=c[L], fmt='.')
        plt.plot(xp, np.ones(len(xp))*eigval[L], color=c[L], linestyle='--', label='$E_{%d}=%f$' %(L, eigval[L]))
        plt.plot(xp, abs(G(xp, L))**2 + eigval[L], color='k')
        
    #plt.legend(loc='best')
    
    plt.figure(2)
    plt.title("$\psi(x)$", fontsize=20)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('$\psi(x)$', fontsize=15)
    plt.grid()

    c = ['b', 'r', 'g']
    
    for L in range(3):
        plt.errorbar(xp, abs(abs(psi[:,L])**2 - abs(G(xp, L))**2), color=c[L], fmt='.')



    plt.show()

