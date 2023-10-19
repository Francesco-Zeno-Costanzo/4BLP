import numpy as np
import matplotlib.pyplot as plt

def Legge_oraria(t, h0, g):
    """
    Restituisce la legge oraria di caduta
    di un corpo che parte da altezza h0 e
    con una velocita' inziale nulla
    """
    return h0 - 0.5*g*t**2

""""
dati misurati:
xdata : fisicamemnte i tempi a cui osservo
        la caduta del corpo non affetti da
        errore
ydata : fisicamente la posizione del corpo
        misurata a dati tempi xdata afetta
        da errore
"""

#misuro 50 tempi tra 0 e 2 secondi
xdata = np.linspace(0, 2, 50)

#legge di caduta del corpo
y = Legge_oraria(xdata, 20, 9.81)
rng = np.random.default_rng()
y_noise = 0.3 * rng.normal(size=xdata.size)
#dati misurati afferri da errore
ydata = y + y_noise
dydata = np.array(ydata.size*[0.3])

N = 100
S2 = np.zeros((N, N))
h0 = np.linspace(15, 25, N)
g = np.linspace(7, 12, N)
for i in range(N):
    for j in range(N):
        S2[i, j] = (((ydata - Legge_oraria(xdata, h0[i], g[j]))/dydata)**2).sum()

#grafico del chi quadro
fig = plt.figure(1)
gridx, gridy = np.meshgrid(h0, g)
ax = fig.add_subplot(projection='3d')
ax.plot_surface(gridx, gridy, S2, color='yellow')
ax.set_title('Chiquadro regressione lineare')
ax.set_xlabel('h0')
ax.set_ylabel('g')
plt.show()