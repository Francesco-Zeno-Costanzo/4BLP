import numpy as np

#array contenente tutti zero
arraydizeri_0 = np.zeros(3)#il numero specificato e' la lunghezza
arraydizeri_1 = np.zeros(3, 'int')

#array contenente tutti uno
arraydiuni_0 = np.ones(5)#il numero specificato e' la lunghezza
arraydiuni_1 = np.ones(5, 'int')

print(arraydizeri_0, arraydizeri_1)
print(arraydiuni_0, arraydiuni_1)

"""
questo invece e' un array il cui primo elemento e' zero
e l'ultimo elemento e' 1, lungo 10 e i cui elementi sono
equispaziati in maniera lineare tra i due estremi
"""
equi_lin = np.linspace(0, 1, 10)
print(equi_lin)


"""
questo invece e' un array il cui primo elemento e' 10^1
e l'ultimo elemento e' 10^2, lungo 10 e i cui elementi sono
equispaziati in maniera logaritmica tra i due estremi
"""
equi_log = np.logspace(1, 2, 10)
print(equi_log)
