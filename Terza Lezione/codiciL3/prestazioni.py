import time
import numpy as np

#inizio a misurare il tempo
start = time.time()


a1 = 0       #variabile che conterra' il risultato
N = int(5e6) # numero di iterazioni da fare = 5 x 10**6

#faccio il conto a 'mano'
for i in range(N):
    a1 += np.sqrt(i)

#finisco di misurare il tempo
end = time.time()-start

print(end)

#inizio a misurare il tempo
start = time.time()

#stesso conto ma fatto tramite le librerie di python
a2 = sum(np.sqrt(np.arange(N)))

#finisco di misurare il tempo
end = time.time()-start


#sperabilmente sarà minore del tempo impegato prima
print(end)