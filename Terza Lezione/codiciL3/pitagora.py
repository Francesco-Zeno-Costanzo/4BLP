import numpy as np

def Area(l, n):
    """
    calcolo area di un poligono
    regolare di lato l e numero lati n
    """
    a = l/2 * 1/np.tan(np.pi/n) #apotema
    p = n*l                     #perimetro
    A = p*a/2                   #area
    return A

l = [3, 4, 5] #dimesioni dei lati, deve essere una terna pitagorica
n = np.arange(4, 12) #numero di lati dei poligoni

A3 = np.array([]) #array in cui ci saranno le aree dei poligoni di lato 3
A4 = np.array([]) #array in cui ci saranno le aree dei poligoni di lato 4
A5 = np.array([]) #array in cui ci saranno le aree dei poligoni di lato 5

for nn in n: #loop sul numero di lati
    for ll in l: #lup sulla dimesione, quindi sul triangolo
        A0 = Area(ll, nn) #calcolo dell'area

        if ll == 3:
            A3 = np.append(A3, A0)
        elif ll == 4:
            A4 = np.append(A4, A0)
        elif ll == 5:
            A5 = np.append(A5, A0)


for i in range(len(n)):
    print(f'A3 + A4 = {A3[i]+A4[i]:.3f}')
    print(f'   A5   = {A5[i]:.3f}')