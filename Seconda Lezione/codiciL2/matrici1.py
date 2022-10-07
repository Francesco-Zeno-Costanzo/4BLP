import numpy as np

#esiste la funzione apposita di numpy per scrivere matrici.
matrice1 = np.matrix('1 2; 3 4; 5 6')
#Si scrivono essenzialmente i vettori riga della matrice separati da ;

#equivalente a:
matrice2 = np.matrix([[1, 2], [3, 4], [5,6]])

print(matrice1)
print(matrice2)


matricedizeri = np.zeros((3, 2)) #tre righe, due colonne: matrice 3x2
print('Matrice di zeri:\n', matricedizeri, '\n')
matricediuni = np.ones((3,2))
print('Matrice di uni:\n', matricediuni, '\n')