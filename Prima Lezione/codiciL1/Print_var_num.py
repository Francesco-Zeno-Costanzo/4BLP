#definiamo una variabile
c = 3.141592653589793

#stampa come intero
print('%d' %c)

#stampa come reale
print('%f' %c) #di default stampa solo prime 6 cifre
print(f'{c}')  #di default stampa tutte le cifre

#per scegliere il numero di cifre, ad esempio sette cifre
print('%.7f' %c)
print(f'{c:.7f}')
