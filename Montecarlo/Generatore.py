import numpy as np

def GEN(r0, n=1, M=2**64, a=6364136223846793005, c=1442695040888963407, norm=True):
    """
    generatore conguenziale lineare
    Parametri
    ---------
    r0 : int
        seed della generazione
    n : int, opzionale
        dimensione lista da generare, di default è 1
    M : int, opzionale
        periodo del generaltore di default è 2**64
    a : int, opzionale
        moltiplicatore del generatore, di default è 6364136223846793005
    c : int, opzionale
        incremento del generatore, di default è 1442695040888963407
    norm : bool, opzionale
        se True il numero restituito è fra zero ed 1

    Returns
    ---------
    r : list
        lista con numeri distribuiti casualmente
    """
    if n==1:
        r = (a*r0 + c)%M
    else:
        r = []
        x = r0
        for i in range(1, n):
            x = (a*x + c)%M
            r.append(x)

    if norm :
        if n==1:
            return float(r)/(M-1)
        else :
            return [float(el)/(M-1) for el in r]
    else :
        return r

if __name__ == '__main__':
    seed = 42

    print(GEN(seed, n=5))
    momenti1 = [np.mean(np.array(GEN(seed, n=int(5e5)))**i) for i in range(1, 10)]
    momenti2 = [1/(1+p) for p in range(1, 10)]

    for M1, M2 in zip(momenti1, momenti2):
        print(f'{M1:.3f}, {M2:.3f}')

