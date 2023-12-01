import numpy as np
import matplotlib.pyplot as plt

xi = 10         # estemo sinisto
xf = -xi        # estemo destro
N  = 1000       # numero punti
dx = (xf-xi)/N  # spaziatura punti

k = 2*np.pi*np.fft.fftfreq(N, dx) # vettore d'onda

#============================== CASO TRANQUILLO ==============================
x = np.linspace(xi, xf, N)                  # array posizioni
f = np.sin(x)*np.exp(-x**2/10)              # funzione di cui calcolare la derivata
g = np.cos(x)*np.exp(-x**2/10) - 2/10*x*f   # derivata analitica

f_hat  = np.fft.fft(f)       # trasformo con fourier
df_hat = 1j*k*f_hat          # moltiplico per l'impulso
df     = np.fft.ifft(df_hat) # derivata nello spazio

plt.figure(1, figsize=(12, 10))

plt.subplot(321)
plt.title('funzione', fontsize=15)
plt.ylabel('f', fontsize=15)
plt.plot(x, f)
plt.grid()

plt.subplot(323)
plt.title('Confronto derivate ', fontsize=15)
plt.ylabel('derivata', fontsize=15)
plt.plot(x, g)
plt.plot(x, df.real)
plt.grid()

plt.subplot(325)
plt.title('differenza fra derivata analitica e numerica', fontsize=15)
plt.ylabel('erorre globale derivata', fontsize=15)
plt.xlabel("X", fontsize=15)
plt.plot(x, abs(g-df))
plt.yscale('log')
plt.grid()
#============================== CASO MENO TRANQUILLO ==============================

def f():
    '''funzione
    '''
    y = []
    for xi in x:
        if xi < -5 :
            y.append(0)
        if xi > -5 and xi < 0:
            y.append(xi + 5)
        if xi > 0 and xi < 5:
            y.append(-xi + 5)
        if xi > 5:
            y.append(0)
    return np.array(y)

def g():
    ''' derivata analitica
    '''
    y = []
    for xi in x:
        if xi < -5 :
            y.append(0)
        if xi > -5 and xi < 0:
            y.append(1)
        if xi > 0 and xi < 5:
            y.append(-1)
        if xi > 5:
            y.append(0)
    return np.array(y)

f = f()
g = g()

f_hat  = np.fft.fft(f)       # trasformo con fourier
df_hat = 1j*k*f_hat          # moltiplico per l'impulso
df     = np.fft.ifft(df_hat) # derivata nello spazio

plt.subplot(322)
plt.title('funzione', fontsize=15)
plt.ylabel('f', fontsize=15)
plt.plot(x, f)
plt.grid()

plt.subplot(324)
plt.title('Confronto derivate ', fontsize=15)
plt.ylabel('derivata', fontsize=15)
plt.plot(x, g)
plt.plot(x, df.real)
plt.grid()

plt.subplot(326)
plt.title('differenza fra derivata analitica e numerica', fontsize=15)
plt.ylabel('erorre globale derivata', fontsize=15)
plt.xlabel("X", fontsize=15)
plt.plot(x, abs(g-df))
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.savefig("cfr_derive_fft.pdf")
plt.show()
