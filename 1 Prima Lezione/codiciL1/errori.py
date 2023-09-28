a = 0
b = 1/a
print(b)

##try except
a = 0
try :
    b = 1/a
except ZeroDivisionError:
    b = 1

print(b)


##raise
"""
leggo un valore da shell
uso del comando try per evitare che venga letto
qualcosa che non sia un numero: e.g. una stringa
"""
try:
    b = int(input('scegliere un valore:'))
except ValueError:
    print('hai digitato qualcosa diverso da un numero, per favore ridigitare')
    b = int(input('scegliere un valore:'))

#se si sbaglia a digitare di nuovo il codice si arresta per ValueError

#controllo se e' possibile proseguire
if b > 7 :
    #se non lo  si blocca il codice sollevando l'eccezione
    messaggio_di_errore = 'il valore scelto risulta insensato in quanto nulla supera 7, misura massima di ogni cosa'
    raise Exception(messaggio_di_errore)
