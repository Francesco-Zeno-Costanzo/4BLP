##while loop

def fattoriale(n):
    """
    Restituisce il fattoriale di un numero
    """
    R = 1
    #finche' e' vero fai ...
    while n > 1:
        R *= n
        n -= 1
    return R

print(fattoriale(5))



##For loop

def fattoriale(n):
    """
    restituisce il fattoriale di un numero
    """
    R = 1
    #finche' i non arriva ad n fai ...
    for i in range(1, n+1):
        R = R*i
    return R

print(fattoriale(5))