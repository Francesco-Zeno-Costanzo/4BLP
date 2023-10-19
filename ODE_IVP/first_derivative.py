"""
Code that compiles some method of calculating a derivative to various orders of accuracy
"""
import numpy as np
import matplotlib.pyplot as plt


def d1b(f, x0, dx):
    '''first order, backward derivative
    '''
    dfdx = (f(x0) - f(x0-dx))/dx
    return dfdx

def d1f(f, x0, dx):
    '''first order, forward derivative
    '''
    dfdx = (f(x0+dx) - f(x0))/dx
    return dfdx

def d2c(f, x0, dx):
    '''second order, symmetric derivative
    '''
    dfdx = (f(x0+dx) - f(x0-dx))/(2.0*dx)
    return dfdx

def d2f(f, x0, dx):
    ''' second order forward derivative
    '''
    dfdx = ( - 3.0*f(x0) + 4.0*f(x0+dx) - f(x0+2.0*dx) )/(2.0*dx)
    return dfdx

def d2b(f, x0, dx):
    ''' second order backward derivative
    '''
    dfdx = ( 3.0*f(x0) - 4.0*f(x0-dx) + f(x0-2.0*dx) )/(2.0*dx)
    return dfdx

def d4c(f, x0, dx):
    ''' fourth order centered derivative
    '''
    dfdx = ( -f(x0+2*dx) + 8.0*f(x0+dx) - 8.0*f(x0-dx) + f(x0-2*dx) )/(12.0*dx)
    return dfdx

#==================================================================
# Plot
#==================================================================

plt.figure(1, figsize=(12, 8))
x = np.linspace(0, 2*np.pi, 100)
f = np.sin
g = np.cos
h = 1e-4
Df = [d1b, d1f, d2c, d2f, d2b, d4c]
l  = ['first order backward', 'first order forward', 'symmetric',
      'second order forward', 'second order backward', 'fourth order central']

for i, df in enumerate(Df):
    plt.subplot(2, 3, i+1)
    plt.title(l[i])
    plt.plot(x, g(x)-np.array([df(f, t, h) for t in x]), 'blue')
    plt.grid()

plt.tight_layout()
plt.show()

#==================================================================
# Plot per vari valori di h
#==================================================================

#array del passo di discretizzazione
h = np.logspace(-15, -1, 1000)
s = ['-', '-', '--', '--', '--', '-.']
c = ['r', 'b', 'k', 'r', 'b', 'k']
plt.figure(2)
plt.title('Errore derivata al variare del passo', fontsize=15)
plt.ylabel('erorre derivata', fontsize=15)
plt.xlabel("grandezza dell'incremento", fontsize=15)

for i, df in enumerate(Df):
    plt.plot(h, abs(df(f, 0.5, h)-g(0.5)), label=l[i], linestyle=s[i], color=c[i])

plt.xscale('log')
plt.yscale('log')
plt.legend(loc='best')
plt.grid()
plt.show()