def f(x, n):
    """
    restituisce la potenza n-esima di un numero x
    Parametri
    ---------
    x, n : float

    Return
    ---------
    v : float
        x**n
    """

    v = x**n

    return v

if __name__ == '__main__':
    #test
    print(f(5, 2))