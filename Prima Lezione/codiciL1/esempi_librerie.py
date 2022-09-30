import numpy

#per usare un contenuto di questa libreria basta scrivere numpy.contenuto
pigreco = numpy.pi
print(pigreco)

#Possiamo anche ribattezzare le librerie in questo modo
import numpy as np
#da ora all'interno del codice numpy si chiama np

eulero = np.e
print(eulero)



import math

coseno=math.cos(0)
seno = math.sin(np.pi/2) #python usa di default gli angoli in radianti!!!
senosbagliato = math.sin(90)

print('Coseno di 0=', coseno, "\nSeno di pi/2=", seno, "\nSeno di 90=", senosbagliato)

#bisogna quindi stare attenti ad avere tutti gli angoli in radianti
angoloingradi = 45
#questa funzione converte gli angoli da gradi a radianti
angoloinradianti = math.radians(angoloingradi)

print("Angolo in gradi:", angoloingradi, "Angolo in radianti:", angoloinradianti)