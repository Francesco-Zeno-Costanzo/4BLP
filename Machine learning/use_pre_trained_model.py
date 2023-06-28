'''
code that takes a model saved on your computer and uses it to make predictions
'''
import joblib
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

modello = joblib.load(r'C:\users\franc\desktop\mod.sav')

#=========================================================
# Creation of data set with same features
#=========================================================

M = 1000 # numer data
N = 200 # len of each curve

X = np.zeros((N, M))          # matrix of featurs
d = np.linspace(0, 1, 5)      # parameter of curvese
t = np.zeros(M)               # target index of d

for i in range(M):

    # random interval
    x1, x2 = np.random.random(2)*5
    # each features is nothing but a curve y=x**k with k element of d
    k = np.random.choice(d)
    X[:, i] = np.linspace(x1, x2, N)**k
    # the target must be integer so we use the corrispective indices
    t[i] = np.where(k==d)[0][0]

#=========================================================
# Prediction on new data
#=========================================================

prediction = modello.predict(X.T)

print(f'Accuracy score = {accuracy_score(t, prediction)}')

skplt.metrics.plot_confusion_matrix(t, prediction)

plt.show()

