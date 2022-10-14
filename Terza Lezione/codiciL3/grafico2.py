import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """
    restituisce il cubo di un numero
    """
    return x**3

def g(x):
    """
    restituisce il quadrato di un numero
    """
    return x**2

#array di numeri equispaziati nel range [-1,1] usiamo:
x = np.linspace(-1, 1, 40)

plt.figure(1) #creiamo la figura

#titolo
plt.title('Grafico funzioni')
#nomi degli assi
plt.xlabel('x')
plt.ylabel('f(x), g(x)')
#plot dei dati
plt.plot(x, f(x), marker='.', linestyle='--', color='blue', label='parabola')
plt.plot(x, g(x), marker='^', linestyle='-', color='red', label='cubica')
#aggiungiamo una leggenda
plt.legend(loc='best')
#aggiungiamo una griglia
plt.grid()
#comando per mostrare a schermo il grafico
plt.show()