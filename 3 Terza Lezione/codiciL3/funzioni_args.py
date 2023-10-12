def molt(*numeri):
    """
    restituisce il prodotto di n numeri
    """
    R = 1
    for numero in numeri:
        R *= numero
    return R

print(molt(2, 7, 10, 11, 42))
print(molt(5, 5))
print(molt(10, 10, 2))