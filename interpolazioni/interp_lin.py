import numpy as np
import matplotlib.pyplot as plt

def f(x, xx, yy):
    """
    restituisce l'interpolazione dei punti xx yy
    x può essere un singolo valore in cui calcolare
    la funzione interpolante o un intero array
    """
    #proviamo se x è un array
    try :
        n = len(x)
        x_in = np.min(xx) <= np.min(x) and np.max(xx) >= np.max(x)
    except TypeError:
        n = 1
        x_in = np.min(xx) <= x <= np.max(xx)

    #se il valore non è nel range corretto è impossibile fare il conto
    if not x_in :
        a = 'uno o più valori in cui calcolare la funzione'
        b = ' interpolante sono fuori dal range di interpolazione'
        errore = a+b
        raise Exception(errore)

    #array che conterrà l'interpolazione
    F = np.zeros(n)

    if n == 1 :
        #controllo dove è la x e trovo l'indice dell'array
        #per sapere in che range bisogna interpolare
        for j in range(len(xx)-1):
            if xx[j] <= x <= xx[j+1]:
                i = j

        A = yy[i] * (xx[i+1] - x)/(xx[i+1] - xx[i])
        B = yy[i+1] * (x - xx[i])/(xx[i+1] - xx[i])
        F[0] = A + B

    else:
        #per ogni valore dell'array in cui voglio calcolare l'interpolazione
        for k, x in enumerate(x):
            #controllo dove è la x e trovo l'indice dell'array
            #per sapere in che range bisogna interpolare
            for j in range(len(xx)-1):
                if xx[j] <= x <= xx[j+1]:
                    i = j

            A = yy[i] * (xx[i+1] - x)/(xx[i+1] - xx[i])
            B = yy[i+1] * (x - xx[i])/(xx[i+1] - xx[i])
            F[k] = A + B

    return F

if __name__ == '__main__':
    x = np.linspace(0, 1, 10)
    y = np.sin(2*np.pi*x)
    z = np.linspace(0, 1, 100)

    plt.figure(1)
    plt.title('Interpolazione lineare')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(z, f(z, x, y), 'b', label='interpolazione')
    plt.plot(x, y, marker='.', linestyle='', c='k', label='dati')
    plt.legend(loc='best')
    plt.grid()
    plt.show()
