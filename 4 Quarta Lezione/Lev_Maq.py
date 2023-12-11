"""
the code performs a linear and non linear regression
Levenberg–Marquardt algorithm. You have to choose
some parameters delicately to make the result make sense
"""

import numpy as np
import matplotlib.pyplot as plt


def lm_fit(func, x, y, x0, sigma=None, tol=1e-6, dense_output=False, absolute_sigma=False):
    """
    Implementation of Levenberg–Marquardt algorithm
    for non-linear least squares. This algorithm interpolates
    between the Gauss–Newton algorithm and the method of
    gradient descent. It is iterative optimization algorithms
    so finds only a local minimum. So you have to be careful
    about the values ​​you pass in x0

    Parameters
    ----------
    f : callable
        fit function
    x : 1darray
        the independent variable where the data is measured.
    y : 1darray
        the dependent data, y <= f(x, {\theta})
    x0 : 1darray
        initial guess
    sigma : None or 1darray
        the uncertainty on y, if None sigma=np.ones(len(y)))
    tol : float
        required tollerance, the algorithm stop if one of this quantities
        R1 = np.max(abs(J.T @ W @ (y - func(x, *x0))))
        R2 = np.max(abs(d/x0))
        R3 = sum(((y - func(x, *x0))/dy)**2)/(N - M) - 1
        is smaller than tol

    dense_output : bool, optional dafult False
        if true all iteration are returned
    absolute_sigma : bool, optional dafult False
        If True, sigma is used in an absolute sense and
        the estimated parameter covariance pcov reflects
        these absolute values.
        pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)

    Returns
    -------
    x0 : 1d array or ndarray
        array solution
    pcov : 2darray
        The estimated covariance of popt
    iter : int
        number of iteration
    """

    iter = 0               #initialize iteration counter
    h = 1e-7               #increment for derivatives
    l = 1e-3               #damping factor
    f = 10                 #factor for update damping factor
    M = len(x0)            #number of variable
    N = len(x)             #number of data
    s = np.zeros(M)        #auxiliary array for derivatives
    J = np.zeros((N, M))   #gradient
    #some trashold
    eps_1 = 1e-1
    eps_2 = tol
    eps_3 = tol
    eps_4 = tol

    if sigma is None :     #error on data
        W  = np.diag(1/np.ones(N))
        dy = np.ones(N)
    else :
        W  = np.diag(1/sigma**2)
        dy = sigma


    if dense_output:       #to store solution
        X = []
        X.append(x0)

    while True:
        #jacobian computation
        for i in range(M):                                  #loop over variables
            s[i] = 1                                        #we select one variable at a time
            dz1 = x0 + s*h                                  #step forward
            dz2 = x0 - s*h                                  #step backward
            J[:,i] = (func(x, *dz1) - func(x, *dz2))/(2*h)  #derivative along z's direction
            s[:] = 0                                        #reset to select the other variables

        JtJ = J.T @ W @ J                         #matrix multiplication, JtJ is an MxM matrix
        dia = np.eye(M)*np.diag(JtJ)              #dia_ii = JtJ_ii ; dia_ij = 0
        res = (y - func(x, *x0))                  #residuals
        b   = J.T @ W @ res                       #ordinate or “dependent variable” values of system
        d   = np.linalg.solve(JtJ + l*dia, b)     #system solution
        x_n = x0 + d                              #solution at new time

        # compute the metric
        chisq_v = sum((res/dy)**2)
        chisq_n = sum(((y - func(x, *x_n))/dy)**2)

        rho = chisq_v - chisq_n
        den = abs( d.T @ (l*np.diag(JtJ)@d + J.T @ W @ res))
        rho = rho/den
        # acceptance
        if rho > eps_1 :         #if i'm closer to the solution
            x0 = x_n             #update solution
            l /= f               #reduce damping factor
        else:
            l *= f               #else magnify

        # Convergence criteria
        R1 = np.max(abs(J.T @ W @ (y - func(x, *x0))))
        R2 = np.max(abs(d/x0))
        R3 = abs(sum(((y - func(x, *x0))/dy)**2)/(N - M) - 1)

        if R1 < eps_2 or R2 < eps_3 or R3 < eps_4:          #break condition
            break

        iter += 1

        if dense_output:
            X.append(x0)

    #compute covariance matrix
    pcov = np.linalg.inv(JtJ)

    if not absolute_sigma:
        s_sq = sum(((y - func(x, *x0))/dy)**2)/(N - M)
        pcov = pcov * s_sq

    if not dense_output:
        return x0, pcov, iter
    else :
        X = np.array(X)
        return X, pcov, iter


if __name__ == "__main__":

    def f(x, m, q):
        """fit function
        """
        return m*np.cos(x*q)

    ##data
    x = np.linspace(1, 5, 27)
    y = f(x, 0.5, 10)
    rng = np.random.default_rng(seed=69420)
    y_noise = 0.1 * rng.normal(size=x.size)
    y  = y + y_noise
    dy = np.array(y.size*[0.1])

    ##fit a mano

    init = np.array([-1, 9.8])   #|
    tau  = 1e-8                  #|> be careful

    pars, covm, iter = lm_fit(f, x, y, init, sigma=dy, tol=tau)
    dpar = np.sqrt(covm.diagonal())
    for i, p, dp in zip(range(len(pars)), pars, dpar):
        print(f"pars{i} = {p:.5f} +- {dp:.5f}")
    print(f"numero di iterazioni = {iter}")

    #Calcoliamo il chi quadro,indice ,per quanto possibile, della bontà del fit:
    chisq = sum(((y - f(x, *pars))/dy)**2.)
    ndof = len(y) - len(pars)
    print(f'chi quadro = {chisq:.3f} ({ndof:d} dof)')

    #Definiamo un matrice di zeri che divverà la matrice di correlazione:
    c=np.zeros((len(pars),len(pars)))
    #Calcoliamo le correlazioni e le inseriamo nella matrice:
    for i in range(0, len(pars)):
        for j in range(0, len(pars)):
            c[i][j] = (covm[i][j])/(np.sqrt(covm.diagonal()[i])*np.sqrt(covm.diagonal()[j]))
    print(c) #matrice di correlazione


    ##Plot
    #Grafichiamo il risultato
    fig1 = plt.figure(1)
    #Parte superiore contenetnte il fit:
    frame1=fig1.add_axes((.1,.35,.8,.6))
    #frame1=fig1.add_axes((trasla lateralmente, trasla verticamente, larghezza, altezza))
    frame1.set_title('Fit dati simulati',fontsize=10)
    plt.ylabel('y [u.a.]',fontsize=10)
    plt.grid()


    plt.errorbar(x, y, dy, fmt='.', color='black', label='dati') #grafico i punti
    t = np.linspace(np.min(x), np.max(x), 10000)
    plt.plot(t, f(t, *pars), color='blue', alpha=0.5, label='best fit') #grafico del best fit
    plt.legend(loc='best')#inserisce la legenda nel posto migliorte


    #Parte inferiore contenente i residui
    frame2=fig1.add_axes((.1,.1,.8,.2))

    #Calcolo i residui normalizzari
    ff = (y - f(x, *pars))/dy
    frame2.set_ylabel('Residui Normalizzati')
    plt.xlabel('x [u.a.]',fontsize=10)

    plt.plot(t, 0*t, color='red', linestyle='--', alpha=0.5) #grafico la retta costantemente zero
    plt.plot(x, ff, '.', color='black') #grafico i residui normalizzati
    plt.grid()


    ##Plot tariettoria
    N = 200
    p1 = np.linspace(-1, 1.25, N)
    p2 = np.linspace(8, 11.5, N)

    S2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            S2[i, j] = (((y - f(x, p1[i], p2[j]))/dy)**2).sum()

    init1 = np.array([-1, 9.8])
    init2 = np.array([-1, 10.6])
    init3 = np.array([-1, 9.4])
    tau   = 1e-8

    popt1, _, _ = lm_fit(f, x, y, init1, sigma=dy, tol=tau, dense_output=True)
    popt2, _, _ = lm_fit(f, x, y, init2, sigma=dy, tol=tau, dense_output=True)
    popt3, _, _ = lm_fit(f, x, y, init3, sigma=dy, tol=tau, dense_output=True)

    plt.figure(2)
    plt.title("Traiettorie soluzioni")
    plt.xlabel('x')
    plt.ylabel('y')
    levels = np.linspace(np.min(S2), np.max(S2), 40)
    p1, p2 = np.meshgrid(p1, p2)
    c=plt.contourf(p1, p2, S2.T , levels=levels, cmap='plasma')
    plt.colorbar(c)
    plt.grid()
    plt.plot(popt1[:,0], popt1[:,1], 'k', label='tariettoria1')
    plt.plot(popt2[:,0], popt2[:,1], 'k', label='tariettoria2')
    plt.plot(popt3[:,0], popt3[:,1], 'k', label='tariettoria3')
    plt.legend(loc='best')

    plt.show()