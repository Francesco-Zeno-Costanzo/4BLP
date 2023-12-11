def assoluto(x):
    """
    restituisce il valore assoluto di un numero
    """
    # se vero restituisci x
    if x >= 0:
        return x
    #altrimenri restituisci -x
    else:
        return -x

print(assoluto(3))
print(assoluto(-3))




def segno(x):
    """
    funzione per capire il segno di un numero
    """
    #se vero ....
    if x > 0:
        return 'Positivo'
    #se invece ....
    elif x == 0:
        return 'Nullo'
    #altrimenti ....
    else:
        return 'Negativo'

print(segno(5))
print(segno(0))
print(segno(-4))