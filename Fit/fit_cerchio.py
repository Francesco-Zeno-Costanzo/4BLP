import numpy as np
import matplotlib.pyplot as plt


def cerchio(xc, yc, r, N, phi_min=0, phi_max=2*np.pi):
    """
    Restituisce un cerchio di centro (xc, yc) e di raggio r
    phi è il parametro di "percorrenza" del cerchio
    """

    phi = np.linspace(phi_min, phi_max, N)

    x = xc + r*np.cos(phi)
    y = yc + r*np.sin(phi)

    return x, y



def fitcerchio(pt, w=None):
    '''
    fit di un cerchio con metodo di coope
    Parameters
    ----------
    pt : 2Darray
        contiene le coordinate del cerchio
    w : None or 1Darray
        w = np.sqrt(dx**2 + dy**2)
        if None => w = np.ones(len(pt[0]))


    Returns
    -----------
    c : 1Darray
        array con le coordinate del centro del cerchio
    r : float
        raggio del cerchio
    d : 1Darray
        array con gli errori associati a c ed r
    A1 : 2Darray
        matrice di covarianza
    '''
    npt = len(pt[0])

    S = np.column_stack((pt.T, np.ones(npt)))
    y = (pt**2).sum(axis=0)

    if w is None:
        w = np.ones(npt)

    w = np.diag(1/w)

    A = S.T @ w @ S #@ è il prodotto matriciale
    b = S.T @ w @ y
    sol = np.linalg.solve(A, b)

    c = 0.5*sol[:-1]
    r = np.sqrt(sol[-1] + c.T @ c)

    d = np.zeros(3)
    A1 = np.linalg.inv(A)

    for i in range(3):
        d[i] = np.sqrt(A1[i,i])
    return c, r, d, A1


if __name__ == "__main__":
    np.random.seed(69420)
    #numero di punti
    N = 50
    #paramentri cerchio
    xc, yc, r1 = 5, -2, 10
    #errori
    ex, ey = 0.5, 0.5
    dy = np.array(N*[ey])
    dx = np.array(N*[ex])
    dr = np.sqrt(dx**2 + dy**2)
    k = np.random.uniform(0, ex, N)
    l = np.random.uniform(0, ey, N)
    #creiamo il cerchio
    x, y = cerchio(xc, yc, r1, N, np.pi/4, 5/3*np.pi)
    x = x + k #aggiungo errore
    y = y + l

    a = np.array([x, y])
    c, r, d , A = fitcerchio(a, dr) #fit

    print(f'x_c = {c[0]:.5f} +- {d[0]:.5f}; valore esatto = {xc:.5f}')
    print(f'y_c = {c[1]:.5f} +- {d[1]:.5f}; valore esatto = {yc:.5f}')
    print(f'r   = {r:.5f} +- {d[2]:.5f}; valore esatto = {r1:.5f}')


    chisq = sum(((np.sqrt((x-c[0])**2 + (y-c[1])**2) - r)/dr)**2.)
    ndof = N - 3
    print(f'chi quadro = {chisq:.3f} ({ndof:d} dof)')

    corr=np.zeros((3,3))
    for i in range(0, 3):
        for j in range(0, 3):
            corr[i][j]=(A[i][j])/(np.sqrt(A.diagonal()[i])*np.sqrt(A.diagonal()[j]))
    print(corr)

    #plot
    fig1 = plt.figure(1, figsize=(7.5,9.3))
    frame1=fig1.add_axes((.1,.35,.8,.6))
    #frame1=fig1.add_axes((trasla lateralmente, trasla verticamente, larghezza, altezza))
    frame1.set_title('Fit dati simulati',fontsize=20)
    plt.ylabel('y [a.u]',fontsize=10)
    plt.grid()

    plt.errorbar(x, y, dy, dx, fmt='.', color='black', label='dati')
    xx, yy = cerchio(c[0], c[1], r, 10000)
    plt.plot(xx, yy, color='blue', alpha=0.5, label='best fit')
    plt.legend(loc='best')


    frame2=fig1.add_axes((.1,.1,.8,.2))
    frame2.set_ylabel('Residui Normalizzati')
    plt.xlabel('x [a.u.]',fontsize=10)

    ff=(np.sqrt((x-c[0])**2 + (y-c[1])**2) - r)/dr
    x1=np.linspace(np.min(x),np.max(x), 1000)
    plt.plot(x1, 0*x1, color='red', linestyle='--', alpha=0.5)
    plt.plot(x, ff, '.', color='black')
    plt.grid()

    plt.show()
