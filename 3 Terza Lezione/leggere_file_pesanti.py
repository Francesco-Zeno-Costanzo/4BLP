import time
import numpy as np

x = np.linspace(0, 1, int(2e6)) # dati

#=====================================================
# file txt
#=====================================================

# salviamo su file
start = time.time()
path = r'c:\Users\franc\Desktop\dati.txt'
f = open(path, 'w')
for i in x:
    f.write(f'{i} \n')
f.close()
end = time.time() - start

print(f"tempo di scrittura txt: {end} s")

#legggiamo da file
start = time.time()
X = np.loadtxt(path, unpack=True)
end = time.time() - start

print(f"tempo di lettura   txt: {end} s")

#=====================================================
# file npy
#=====================================================

# salviamo su file
start = time.time()
path = r'c:\Users\franc\Desktop\dati.npy'
np.save(path, x)
end = time.time() - start

print(f"tempo di scrittura npy: {end} s")

#legggiamo da file
start = time.time()
X = np.load(path, allow_pickle='TRUE')
end = time.time() - start

print(f"tempo di lettura   npy: {end} s")