"""
liste tuple e dizionari sono ogetti indicizzati che possono
ogni tipo di informazioni al loro interno
"""
lista = [0, 1, 'lampada', [0, 23], (29, 11), {3:4, 'capra':'panca'}] # lista
tupla = (3, 2, "ruspa", [0, 0], (3, 9), {87:90})
dictz = {0:1, 1:[2, 3], 'astolfo':(2, 3), "diz":{1:2}}

# stampo tutto a schermo
print(lista)
print(tupla)
print(dictz)

# tramte l'indice accedo all'elemento della lista o della tupla
print(lista[3])
print(tupla[3])

# per il dizionario va invece usata la chiave
print(dictz['astolfo'])
print(dictz[1])

# modifico elemento all'indice zero nella lista e nel dizionario
lista[0] = (0, 1, 2, 3, 4, 5)
dictz[0] = "BUONA PASQUA"

print(lista)
print(dictz)

# se lo facessi con la tupla avrei un errore
tupla[0] = 1
