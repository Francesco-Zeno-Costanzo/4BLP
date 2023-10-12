##enumerate
import numpy as np

#creiamo un array
array = np.linspace(0, 1, 5)

"""
in questo modo posso iterare contemporaneamente
sia sugli indici sia sugli elementi dell'array
"""
for index, elem in enumerate(array):
    print(index, elem)


##zip

import numpy as np

#creiamo tre un array
array1 = np.linspace(0, 1, 5)
array2 = np.linspace(1, 2, 5)
array3 = np.linspace(2, 3, 5)
"""
in questo modo posso iterare contemporaneamente
sugli elementi di tutti gli array
"""
for a1, a2, a3 in zip(array1, array2, array3):
    print(a1, a2, a3)