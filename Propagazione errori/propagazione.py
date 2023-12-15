import numpy as np
import sympy as sp

x = sp.Symbol('x')
y = sp.Symbol('y')
z = sp.Symbol('z')
t = sp.Symbol('t')

def Errore(x1, dx1, y1, dy1, z1, dz1, t1, dt1):
    """
    Prende in input certe quantità con un errore
    e propaga l'errore su una certa funzione di queste
    """
    #funzione su cui propagare l'errore da modifica all'occorenza
    f1 = ((x-y)/(z+t))

    #valor medio
    f = float(f1.subs(x,x1).subs(y,y1).subs(z,z1).subs(t,t1))

    #derivate parziali calcolate nel punto
    a = sp.diff(f1, x).subs(x,x1).subs(y,y1).subs(z,z1).subs(t,t1)
    b = sp.diff(f1, y).subs(x,x1).subs(y,y1).subs(z,z1).subs(t,t1)
    c = sp.diff(f1, z).subs(x,x1).subs(y,y1).subs(z,z1).subs(t,t1)
    d = sp.diff(f1, t).subs(x,x1).subs(y,y1).subs(z,z1).subs(t,t1)

    #somma dei vari contributi
    df1 = ((a*dx1)**2 + (b*dy1)**2+ (c*dz1)**2 + (d*dt1)**2 )
    df = np.sqrt(float(df1))

    return f, df


print(Errore(1, 0.1, 2, 0.1, 3, 0.1, 2, 0.1))