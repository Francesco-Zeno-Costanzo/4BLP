import numpy as np
import matplotlib.pyplot as plt


def f(x) :
    """
    funzione di cui trovare lo zero
    """
    return 5.0+4.0*x-np.exp(x)

a = 0.0 #estemo sinistro dell'intervallo
b = 4.0 #estremo destro dell'intervallo
t = 1.0e-15 #tolleranza

x=np.linspace(a, b, 1000)
#plot per vedere come scegliere gli estremi
plt.figure(1)
plt.plot(x, f(x))
plt.grid()
plt.show()

##metodo bisezione
fa = f(a)
fb = f(b)
if fa*fb>0:
    print("protrebbero esserci più soluzioni" , fa , fb)
"""
Potrebbero esserci più zeri anche se la condizione non fosse verificata
Ma se la condizione è verificata allora di certo ci sono piu' soluzioni
non e' un se e solo se
"""

iter = 1
#fai finche' l'intervallo e' piu' grande della tolleranza
while (b-a) > t:
    c = (a+b)/2.0 #punto medio
    fc = f(c)
    #se hanno lo stesso segno allora c è piu' vicino allo zero che a
    if fc*fa > 0:
        a = c
    #altrimenti e' b che è piu' lontano
    else:
        b = c
    iter += 1

print(iter , " iterazioni necessarie:")
print("x0 = " ,c)
print("accuracy = " , '{:.2e}' .format(b-a))
print("f (x0)=" ,f(c))

##Costuzione metodo

a = 0.0 #estemo sinistro dell'intervallo
b = 4.0 #estremo destro dell'intervallo
t = 1.0e-15 #tolleranza

plt.figure(2)
plt.title('Costuzione metodo di bisezione')
plt.plot(x, f(x), 'b')
plt.plot([a, b],[f(a), f(b)], linestyle='--', c='r')
plt.plot(x, x*0, 'k')

iter = 1
#fai finche' l'intervallo e' piu' grande della tolleranza
while (b-a) > t:
    c = (a+b)/2.0 #punto medio
    fc = f(c)
    #se hanno lo stesso segno allora c è piu' vicino allo zero che a
    if fc*fa > 0:
        a = c
    #altrimenti e' b che è piu' lontano
    else:
        b = c
    iter += 1
    plt.plot([a, b],[f(a), f(b)], linestyle='--', c='r')

plt.grid()
plt.show()
