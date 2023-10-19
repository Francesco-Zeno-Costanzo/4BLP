'''
code that trains a model and saves it
'''
import joblib
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

path = r'C:\Users\franc\desktop\mod.sav'

#=========================================================
# Creation of data set
#=========================================================

M = 30000 # numer data
N = 200   # len of each curve

X = np.zeros((N, M))          # matrix of featurs
d = np.linspace(0, 1, 5)      # parameter of curvese
t = np.zeros(M)               # target index of d

for i in range(M):

    # random interval
    x1, x2 = np.random.random(2)*5
    # each features is nothing but a curve y=x**k with k element of d
    k = d[i%len(d)]
    X[:, i] = np.linspace(x1, x2, N)**k
    # the target must be integer so we use the corrispective indices
    t[i] = i%len(d)

#=========================================================
# Creation and training of model
#=========================================================

x = X.T
y = t
# split fro train ad test
x_train, x_test, y_train, y_test = train_test_split(x, y)

# define the model
modello = DecisionTreeClassifier()
# I train the model
modello.fit(x_train, y_train)
# predictions about the data on which it has learned
prediction_train = modello.predict(x_train)
# prediction on new data
prediction_test  = modello.predict(x_test)

# accuracy
print("accuracy train")
print(accuracy_score(y_train, prediction_train))

print("accuracy test")
print(accuracy_score(y_test, prediction_test))

#=========================================================
# Plot confusion matrix
#=========================================================

skplt.metrics.plot_confusion_matrix(y_train, prediction_train)
skplt.metrics.plot_confusion_matrix(y_test,  prediction_test)

plt.show()

#=========================================================
# Save model
#=========================================================

joblib.dump(modello, path)