import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Legge_oraria(t, A, omega):
    """
    Restituisce la legge oraria di un corpo che
    oscilla con ampiezza A e frequenza omega
    """
    return A*np.cos(omega*t)

""""
dati misurati:
xdata : fisicamemnte i tempi a cui osservo
        l'osscilazione del corpo non
        affetti da errore
ydata : fisicamente la posizione del corpo
        misurata a dati tempi xdata afetta
        da errore
"""

#misuro 50 tempi tra 0 e 2 secondi
xdata = np.linspace(0, 2, 50)

#legge di oscillazione del corpo
y = Legge_oraria(xdata, 10, 42)
rng = np.random.default_rng()
y_noise = 0.3 * rng.normal(size=xdata.size)
#dati misurati afferri da errore
ydata = y + y_noise
dydata = np.array(ydata.size*[0.3])

N = 100
S2 = np.zeros((N, N))
A = np.linspace(5, 15, N)
O = np.linspace(30, 50, N)
for i in range(N):
    for j in range(N):
        S2[i, j] = (((ydata - Legge_oraria(xdata, A[i], O[j]))/dydata)**2).sum()

#grafico chiquadro
fig = plt.figure(1)
gridx, gridy = np.meshgrid(A, O)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(gridx, gridy, S2, color='yellow')
ax.set_title('Chiquadro regressione non-lineare')
ax.set_xlabel('A')
ax.set_ylabel('Omega')
plt.show()