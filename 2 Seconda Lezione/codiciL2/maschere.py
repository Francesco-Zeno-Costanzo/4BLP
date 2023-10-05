import numpy as np

# array che contiene valori logici, detti booleani
b = np.array([True, False])

print(b)

# normalissimo array numerico
x = np.array([32, 89])
y = x[b] # x in corrispondenza di indici di b
print(y) 
