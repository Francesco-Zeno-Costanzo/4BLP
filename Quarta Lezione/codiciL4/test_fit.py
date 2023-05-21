"""
Test
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Lev_Maq import lm_fit


def f(t, A, o1, o2, f1, f2, v, tau):
    """fit function
    """
    return A*np.cos(t*o1 + f1)*np.cos(t*o2 + f2)*np.exp(-t/tau) + v

##data
x = np.linspace(0, 20, 1000)
y = f(x, 200, 10.5, 0.5, np.pi/2, np.pi/4, 42, 25)
rng = np.random.default_rng(seed=69420)
y_noise = 1 * rng.normal(size=x.size)
y  = y + y_noise
dy = np.array(y.size*[1])

##confronto

init = np.array([101, 10.5, 0.475, 1.5, 0.6, 35, 20])

pars1, covm1, iter = lm_fit(f, x, y, init, sigma=dy, tol=1e-8)
dpar1 = np.sqrt(covm1.diagonal())
pars2, covm2 = curve_fit(f, x, y, init, sigma=dy)
dpar2 = np.sqrt(covm2.diagonal())
print("        ------codice---------|--------scipy----------")
for i, p1, dp1, p2, dp2 in zip(range(len(init)), pars1, dpar1, pars2, dpar2):
    print(f"pars{i} = {p1:.5f} +- {dp1:.5f} ; {p2:.5f} +- {dp2:.5f}")

print(f"numero di iterazioni = {iter}")

#Calcoliamo il chi quadro,indice ,per quanto possibile, della bontà del fit:
chisq1 = sum(((y - f(x, *pars1))/dy)**2.)
chisq2 = sum(((y - f(x, *pars2))/dy)**2.)
ndof = len(y) - len(pars1)
print(f'chi quadro codice = {chisq1:.3f} ({ndof:d} dof)')
print(f'chi quadro numpy  = {chisq2:.3f} ({ndof:d} dof)')

#Definiamo un matrice di zeri che divverà la matrice di correlazione:
c1 = np.zeros((len(pars1),len(pars1)))
c2 = np.zeros((len(pars1),len(pars1)))
#Calcoliamo le correlazioni e le inseriamo nella matrice:
for i in range(0, len(pars1)):
    for j in range(0, len(pars1)):
       c1[i][j] = (covm1[i][j])/(np.sqrt(covm1.diagonal()[i])*np.sqrt(covm1.diagonal()[j]))
       c2[i][j] = (covm2[i][j])/(np.sqrt(covm2.diagonal()[i])*np.sqrt(covm2.diagonal()[j]))
#print(c1) #matrice di correlazione
#print(c2)

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
plt.plot(t, f(t, *pars1), color='blue', alpha=0.5, label='best fit codice') #grafico del best fit
plt.plot(t, f(t, *pars2), color='red' , alpha=0.5, label='best fit scipy') #grafico del best fit scipy
plt.legend(loc='best')#inserisce la legenda nel posto migliorte

#Parte inferiore contenente i residui
frame2=fig1.add_axes((.1,.1,.8,.2))

#Calcolo i residui normalizzari
ff1 = (y - f(x, *pars1))/dy
ff2 = (y - f(x, *pars2))/dy
frame2.set_ylabel('Residui Normalizzati')
plt.xlabel('x [u.a.]',fontsize=10)

plt.plot(t, 0*t, color='red', linestyle='--', alpha=0.5) #grafico la retta costantemente zero
plt.plot(x, ff1, '.', color='blue') #grafico i residui normalizzati
plt.plot(x, ff2, '.', color='red') #grafico i residui normalizzati scipy
plt.grid()
plt.show()