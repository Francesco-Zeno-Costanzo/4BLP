import numpy as np

matrice1 = np.matrix('1 2; 3 4; 5 6')
matricediuni = np.ones((3,2))

sommadimatrici = matrice1 + matricediuni
print('Somma di matrici:\n', sommadimatrici)

matrice3 = np.matrix('3 4 5; 6 7 8') #matrice 2x3
prodottodimatrici = matrice1 * matrice3  #matrice 3x(2x2)x3
#alternativamente si potrebbe scrivere: prodottodimatrici = matrice1 @ matrice3

print('\nProdotto di matrici:\n', prodottodimatrici)
