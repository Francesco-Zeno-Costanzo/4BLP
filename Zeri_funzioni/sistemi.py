import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root

## risoluzione "manuale"
x = sp.Symbol('x')
y = sp.Symbol('y')

f1 = x**2 + y**2 - 1
f2 = y - x**2 + x/2

f1x = sp.diff(f1, x)
f1y = sp.diff(f1, y)
f2x = sp.diff(f2, x)
f2y = sp.diff(f2, y)

#valori iniziali da cui partire che vengono
#ongi volta aggiornati fino ad arrivare alla soluzione
x0 = 0.2
y0 = 0.0

tau = 1e-10
iter = 0
xs = np.array([x0])
ys = np.array([y0])

while (abs(f1.subs(x, x0).subs(y, y0))>tau and abs(f2.subs(x, x0).subs(y, y0)) > tau):
    #calcolo jacobiano
    a11 = f1x.subs(x, x0).subs(y, y0)
    a12 = f2x.subs(x, x0).subs(y, y0)
    a21 = f1y.subs(x, x0).subs(y, y0)
    a22 = f2y.subs(x, x0).subs(y, y0)

    det_a = a11*a22 - a12*a21
    #calcolo inverso
    b11 = a22/det_a
    b12 = -a21/det_a
    b21 = -a12/det_a
    b22 = a11/det_a

    f1_0 = f1.subs(x, x0).subs(y, y0)
    f2_0 = f2.subs(x, x0).subs(y, y0)
    #risolvo
    d1 = -(b11*f1_0 + b12*f2_0)
    d2 = -(b21*f1_0 + b22*f2_0)
    #aggiorno le coordinate
    x0 = x0 + d1
    y0 = y0 + d2
    #conservo per fare il plot
    xs = np.insert(xs, len(xs), x0)
    ys = np.insert(ys, len(ys), y0)

    iter += 1

print(f"x_0: {xs[-1]} e y_0: {ys[-1]} raggiunti in {iter} iterazioni")

t = np.linspace(0, 2*np.pi, 1000)
z = np.linspace(-0.8, 1.5, 1000)

plt.figure(1)
plt.grid()
plt.plot(np.cos(t), np.sin(t), 'r', label='prima equazione')
plt.plot(z, z**2- z/2, 'b', label='seconda equazione')
plt.plot(xs, ys, 'k', label='evoluzione della soluzione')
plt.legend(loc='best')
plt.show()

##risoluzione con fsolve di scipy

def sistema(V):
    x1, x2 = V
    r1 = x1**2 + x2**2 - 1
    r2 = x2 - x1**2 + x1/2
    return[r1 , r2]

start = (0.2, 0.0)
sol = fsolve(sistema , start, xtol=1e-10)
print("soluzione con fsolve:", sol)

##risoluzione con root di scipy

def sistema(V):
    x1, x2 = V
    r1 = x1**2 + x2**2 - 1
    r2 = x2 - x1**2 + x1/2
    return[r1 , r2]

start = (0.2, 0.0)
sol = root(sistema, start, method='hybr')
print("soluzione con root:", sol.x)