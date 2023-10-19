import numpy as np
import matplotlib.pyplot as plt


def ellisse(parametri, n, tmin=0, tmax=2*np.pi):
    """
    Resistuisce un'ellisse di centro (x0, y0),
    di semiassi maggiore e minore (semi_M, semi_m)
    inclinata di un anglo (phi) rispetto all'asse x
    t è il parametro di "percorrenza" dell'ellisse
    """

    x0, y0, semi_M, semi_m, phi = parametri
    t = np.linspace(tmin, tmax, n)

    x = x0 + semi_M*np.cos(t)*np.cos(phi) - semi_m*np.sin(t)*np.sin(phi)
    y = y0 + semi_M*np.cos(t)*np.sin(phi) + semi_m*np.sin(t)*np.cos(phi)

    return x, y


def cartesiano_a_polari(coef):
    """
    Converte i coefficenti di: ax^2 + bxy + cy^2 + dx + fy + g = 0
    nei coefficeniti polari: centro, semiassi, inclinazione ed eccentricità
    Per dubbi sulla geometria: https://mathworld.wolfram.com/Ellipse.html
    """
    #i termini misti presentano un 2 nella forma più generale
    a = coef[0]
    b = coef[1]/2
    c = coef[2]
    d = coef[3]/2
    f = coef[4]/2
    g = coef[5]

    #Controlliamo sia un ellisse (i.e. il fit sia venuto bene, forse)
    den = b**2 - a*c
    if den > 0:
        Error = 'I coefficenti passati non sono un ellisse: b^2 - 4ac deve essere negativo'
        raise ValueError(Error)

    #Troviamo il centro dell'ellisse
    x0, y0 = (c*d - b*f)/den, (a*f - b*d)/den

    num = 2*(a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    #Troviamo i semiassi maggiori e minori
    semi_M = np.sqrt(num/den/(fac - a - c))
    semi_m = np.sqrt(num/den/(-fac - a - c))

    #Controlliamo che il semiasse maggiore sia maggiore
    M_gt_m = True
    if semi_M < semi_m:
        M_gt_m = False
        semi_M, semi_m = semi_m, semi_M

    #Troviamo l'eccentricità
    r = (semi_m/semi_M)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    #Troviamo l'angolo di inclinazione del semiasse maggiore dall'asse x
    #l'angolo come solito e misurato in senso antiorario
    if b == 0:
        if a < c :
            phi = 0
        else:
            phi = np.pi/2

    else:
        phi = np.arctan((2*b)/(a - c))/2
        if a > c:
            phi += np.pi/2

    if not M_gt_m :
        phi += np.pi/2

    #periodicità della rotazione
    phi = phi % np.pi

    return x0, y0, semi_M, semi_m, e, phi


def fit_ellisse(x, y):
    """
    Basato sull'articolo di Halir and Flusser,
    "Numerically stable direct
    least squares fitting of ellipses".
    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T

    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M

    eigval, eigvec = np.linalg.eig(M)
    cond = 4*eigvec[0]*eigvec[2] - eigvec[1]**2
    ak = eigvec[:, cond > 0]

    return np.concatenate((ak, T @ ak)).ravel()


if __name__ == "__main__":

    #numero di punti
    N = 100
    #parametri dell'ellissi
    x0, y0 = 4, -3.5
    semi_M, semi_m = 7, 3
    phi = np.pi/4
    #eccentricità non fondamentale per la creazione
    r = (semi_m/semi_M)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    #errori
    ex, ey = 0.2, 0.2
    dy = np.array(N*[ey])
    dx = np.array(N*[ex])
    #creiamo l'ellisse
    x, y = ellisse((x0, y0, semi_M, semi_m, phi), N, np.pi/4, 3/2*np.pi)
    k = np.random.uniform(0, ex, N)
    l = np.random.uniform(0, ey, N)
    x = x + k #aggiungo errore
    y = y + l

    coef_cart = fit_ellisse(x, y) #fit

    print('valori esatti:')
    print(f'x0:{x0:.4f}, y0:{y0:.4f}, semi_M:{semi_M:.4f}, semi_m:{semi_m:.4f}, phi:{phi:.4f}, e:{e:.4f}')
    x0, y0, semi_M, semi_m, e, phi = cartesiano_a_polari(coef_cart)
    print('valori fittati')
    print(f'x0:{x0:.4f}, y0:{y0:.4f}, semi_M:{semi_M:.4f}, semi_m:{semi_m:.4f}, phi:{phi:.4f}, e:{e:.4f}')

    #plot
    plt.figure(1)
    plt.title('Fit dati simulati',fontsize=20)
    plt.ylabel('y [a.u]',fontsize=10)
    plt.xlabel('x [a.u]',fontsize=10)
    plt.axis('equal')
    plt.errorbar(x, y, dy, dx, fmt='.', color='black', label='dati')
    x, y = ellisse((x0, y0, semi_M, semi_m, phi), 1000)
    plt.plot(x, y)
    plt.grid()
    plt.show()
