import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Importiamo i dati (va inserito il path assoluto per permettere di trovare) e definiamo la funzione di fit:
#x, y= np.loadtxt(r'C:\Users\franc\Desktop\datiL\DatiL2\onda.txt', unpack = True)
N = 500
ex, ey = 0.1, 1
dy = np.array(N*[ey])
dx = np.array(N*[ex])
x = np.linspace(0, 50, N)

A1 = 20
o1 = 2
v1 = 30
phi = np.pi/4

y = A1*np.sin(o1*x + phi) + v1
k = np.random.uniform(0, ey, N)
l = np.random.uniform(0, ex, N)
y = y + k #aggiungo errore
x = x + l

def f(x, A, o, f, v):
    '''funzione modello
    '''
    return A*np.sin(o*x + f) + v

"""
definiamo un array di parametri iniziali contenente
i volori numerici che ci si aspetta il fit restituisca,
per aiutare la convergenza dello stesso:
init = np.array([A, o, f, v])
"""
init = np.array([25, 2.1, 3, 29])


#Eseguiamo il fit e stampiamo i risultati:
pars, covm = curve_fit(f, x, y, init, sigma=dy, absolute_sigma=False)
print('A  = %.5f +- %.5f ' % (pars[0], np.sqrt(covm.diagonal()[0])))
print('o  = %.5f +- %.5f ' % (pars[1], np.sqrt(covm.diagonal()[1])))
print('f  = %.5f +- %.5f ' % (pars[2], np.sqrt(covm.diagonal()[2])))
print('v  = %.5f +- %.5f ' % (pars[3], np.sqrt(covm.diagonal()[3])))

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


#Grafichiamo il risultato
fig1 = plt.figure(1)
#Parte superiore contenetnte il fit:
frame1=fig1.add_axes((.1,.35,.8,.6))
#frame1=fig1.add_axes((trasla lateralmente, trasla verticamente, larghezza, altezza))
frame1.set_title('Fit dati simulati',fontsize=20)
plt.ylabel('ampiezza [u.a.]',fontsize=10)
#plt.ticklabel_format(axis = 'both', style = 'sci', scilimits = (0,0))#notazione scientifica sugliassi
plt.grid()


plt.errorbar(x, y, dy, dx, fmt='.', color='black', label='dati') #grafico i punti
t = np.linspace(np.min(x),np.max(x), 10000)
s = f(t, *pars)
plt.plot(t,s, color='blue', alpha=0.5, label='best fit') #grafico del best fit
plt.legend(loc='best')#inserisce la legenda nel posto migliorte


#Parte inferiore contenente i residui
frame2=fig1.add_axes((.1,.1,.8,.2))

#Calcolo i residui normalizzari
ff = (y-f(x, *pars))/dy
frame2.set_ylabel('Residui Normalizzati')
plt.xlabel('tempo [u.a.]',fontsize=10)
#plt.ticklabel_format(axis = 'both', style = 'sci', scilimits = (0,0))


plt.plot(t, 0*t, color='red', linestyle='--', alpha=0.5) #grafico la retta costantemente zero
plt.plot(x, ff, '.', color='black') #grafico i residui normalizzati
plt.grid()

plt.show()