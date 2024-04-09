"""
Codice Per plottare i dati con una funzione per capire
i valori dei parametri ottimali da passare a curve_fit
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

# Leggo i dati
#x_data, y_data, dy_data = np.loadtxt('...', unpack=True)

# Qui per comodità li simulo
x_data  = np.linspace(0, 10, 60)
y_data  = 10*np.cos(2.5*x_data + np.pi/4) + 3

# Un po' di rumore quanto basta
rng = np.random.default_rng(seed=69420)
dy      = 1
y_noise = dy * rng.normal(size=x_data.size)
y_data +=  y_noise
dy_data = np.array(x_data.size*[dy])

# Creazione della figura
plt.figure(figsize=(8, 8))
plt.title("TITOLO")
plt.xlabel("t", fontsize=15)
plt.ylabel("F(t)", fontsize=15)
plt.subplots_adjust(bottom=0.2)
plt.errorbar(x_data, y_data, dy_data, c='k', fmt='.', label='data')

# Testo da scrivere inizialemente sulla barra per spiegare
text = "Insert here function, e.g. np.cos(t) or 3*t - 2 then press enter"

# Linspace per il plot e definiamo la variabile l che è l'output del grafico
# ci servirà in quanto noi andremo a sovrascrivere questa variabile in modo
# che sia sempre tutto associato a questo grafico. Di default si plotta una
# retta alla media dei dati
t  = np.linspace(np.min(x_data), np.max(x_data), 1000)
l, = plt.plot(t, np.mean(y_data)*np.ones(t.size), 'b', label='fit law')
plt.legend(loc='best')
plt.grid()

def submit(text):
    '''
    Funzione che valuta l'espressione e la plotta

    Parameter
    ---------
    text : string
        Espressione da valutare scritta in python
    '''
    ydata = eval(text) # valuto l'espressione
    l.set_ydata(ydata) # aggiorno la variabile del plot
    plt.draw()         # Disegno il plot aggiornato

# Box per prendere l'input
axbox = plt.axes([0.15, 0.05, 0.75, 0.075])
text_box = TextBox(axbox, 'F(t)=', initial=text)
text_box.on_submit(submit)

plt.show()
