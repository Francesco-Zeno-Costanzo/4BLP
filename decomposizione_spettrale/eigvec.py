import time
import numpy as np
import scipy.special as ssp
from scipy.sparse import diags
import matplotlib.pyplot as plt


def eig(M, k=None, tol=1e-10, magnitude='small'):
    '''
    Compute the eigenvalue decomposition
    of the symmetric matrix A using power iteration
    or inverse iteration.
    Inverse iteration is the same of power iteration
    but we use M^-1 instead of M so the eigenvalues
    are the reciprocal.

    Parameters
    ----------
    M : 2darray
        N x N matrix, symmetric
    k : None or int, if None k=N
        number of eigenvariates and eigenvectors to find,
        if k<N then k eigenvectors corresponding to the k
        largest eigenvalues ​​will be found
    tol : float, optional default 1e-10
        required tollerance
    magnitude : string, optional, default small
        if magnitude == 'small' the smallest eigenvalues
        and thei relative eigenvectors will be computed
        if magnitude == 'big' the biggest eigenvalues
        and thei relative eigenvectors will be computed

    Return
    ------
    eigval : 1darray
        array of eigenvalues
    eigvec : 2darray
        kxk matrix, the column eigvec[:, i] is the
        ormalized eigenvector corresponding to the
        eigenvalue eigval[i]
    counts : 1darray
        how many iteration are made for each eigenvector
    '''

    if magnitude == 'small':
        A = np.copy(np.linalg.inv(M))
    if magnitude == 'big':
        A = np.copy(M)
        
    N = A.shape[0]
    if k is None:
        k = N

    eigvec = []  # will contain the eignvectors
    eigval = []  # will contain the eignvalues
    counts = []  # will contain the numbero of iteration of each eigenvalue

    for _ in range(k):

        v_p = np.random.randn(N) #initial vector
        v_p = v_p / np.sqrt(sum(v_p**2))
        l_v = np.random.random()
        Iter= 0

        while True:
            l_o = l_v
            v_o = v_p                          # update vector
            v_p = np.dot(A, v_p)               # compute new vector
            v_p /= np.sqrt(sum(v_p**2))        # normalization
            v_p = gs(v_p, eigvec)              # orthogonalization respect
                                               # all eigenvectors find previously
            #eigenvalue of v_p, A @ v_p = l_v * v_p
            #multiplying by the transposed => (A @ v_p) @ v_p.T = l_v
            #using v_p @ v_p.T = 1
            l_v = np.dot(np.dot(A, v_p), v_p)
            
            R1 = np.sqrt(sum((v_p - v_o)**2))
            R2 = np.sqrt(sum((v_o + v_p)**2))
            R3 = np.sqrt(abs(l_v - l_o))      # In eigenvalues the convergence is quadratic

            Iter += 1
            if R1 < tol or R2 < tol or R3 < tol:
                break

        eigvec.append(v_p)
        eigval.append(l_v)
        counts.append(Iter)
    
    if magnitude == 'small':
        eigval = 1/np.array(eigval)
        eigvec = np.array(eigvec).T
    if magnitude == 'big':
        eigvec = np.array(eigvec).T
        eigval = np.array(eigval)

    return eigvec, eigval, counts


def gs(v, eigvec):
    """
    Gram–Schmidt process for orthogonalization

    Parameters
    ----------
    v : 1darray
        vector to be orthogonalized
    eigvec : list
        list of normalized eigenvectors, vectors
        with respect to which to orthogonalize v

    Return
    ------
    v : 1darray
        vector orthogonal to the eigenvector set
    """

    for i in range(len(eigvec)):
        v = v - np.dot(eigvec[i], v) * eigvec[i]
    return v
    

def test(n, k, mag):
    

    P = np.random.normal(size=[n, n])
    H = np.dot(P.T, P)  #for symmetric matrix
    
    start = time.time()
    eigvec1, eigval1, Iter1 = eig(H, k, tol=1e-10, magnitude=mag)
    end = time.time() - start
    print(f'Elapsed time       = {end:.5f}')
    
    start = time.time()
    eigval2, eigvec2 = np.linalg.eigh(H)
    end = time.time() - start
    print(f'Elapsed time numpy = {end:.5f}') 
    
    if mag =='small' : 
        eigvals = np.sort(eigval2)             # for same notation
        eigvecs = eigvec2[:,eigval2.argsort()] # using in eig
        diff_l1 = abs(eigval1-eigvals[:k])
        
    if mag =='big' :
        eigvals = np.sort(eigval2)[::-1]             # for same notation
        eigvecs = eigvec2[:,eigval2.argsort()[::-1]] # using in eig
        diff_l1 = abs(eigval1-eigvals[:k])
    
    diff_n1 = np.zeros(k)
    
    for i in range(k):

        ps = np.dot(eigvec1[:,i], eigvecs[:,i])
        if ps < 0:      #if v is an eigenvector
            diff_n1[i] = np.sqrt(sum((eigvec1[:,i]+eigvecs[:,i])**2))
        else :          #also -v is an eigenvectors
            diff_n1[i] = np.sqrt(sum((eigvec1[:,i]-eigvecs[:,i])**2))
    
    plt.figure(0)

    plt.subplot(311)
    plt.title('differce between eigenvalues')
    plt.ylabel(r'$abs( \lambda - \lambda_{numpy})$')
    plt.yscale('log')
    plt.grid()
    plt.plot(diff_l1, marker='.', linestyle='', color='blue')

    plt.subplot(312)
    plt.title('differce between eigenvectors')
    plt.ylabel(r'$|\vec{v} - \vec{v}_{numpy}|$')
    plt.yscale('log')
    plt.grid()
    plt.plot(diff_n1, marker='.', linestyle='', color='blue')
    
    plt.subplot(313)
    plt.title('Number of iteration required')
    plt.ylabel('iteration$')
    plt.xlabel('number')
    plt.yscale('log')
    plt.grid()
    
    plt.plot(Iter1, 'b', label='power')



if __name__ == "__main__":

    np.random.seed(69420)
    
    test(100, 100, 'big')
    
#===============================================================================
# Computational parameter
#===============================================================================
         
    k  = 10                    # how many levels compute
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
    H = -(1/(2*h**2))*P + V

#===============================================================================
# Computation
#===============================================================================
    
    start = time.time()
    eigvec, eigval, Iter = eig(H, k, tol=1e-5, magnitude='small')
    end = time.time() - start
    print(f'Elapsed time       = {end:.5f}\n')
    
    print("Theoretical     Computed          error")
    print("-----------------------------------------")
    for i in range(k):
        print(f'{i + 0.5} \t \t {eigval[i]:.5f} \t {eigval[i]-(i+0.5):.2e}')
    
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
    
