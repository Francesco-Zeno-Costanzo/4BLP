import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

"""
utilizzo di un regressore lineare per predirre
Il prrezzo di una casa dati certe informazioni
"""
dataset = load_boston()

print(dataset["DESCR"])
#primi 13 elemnti della prima riga
print(dataset["data"][0])
#ultimo elemnto della prima riga
print(dataset["target"][0])

#caratteristiche
X = dataset["data"]

#output, cioe' quello che il modello dovrebbe predire
y = dataset["target"]

#divido i dati, un parte li uso per addestrare, l'altra per
#testare se il modello ha imparato bene
X_train, X_test, y_train, y_test = train_test_split(X, y)

#modello non addrestato e' un regressore
#un regressore prende i dati e restituisce un numero, una stima di qualcosa
modello = LinearRegression()

#addestro il modello
modello.fit(X_train, y_train)

#predizioni
p_train = modello.predict(X_train)
p_test = modello.predict(X_test)

#errori sulla predizione
dp_train = mean_absolute_error(y_train, p_train)
dp_test = mean_absolute_error(y_test, p_test)

print("train", np.mean(y_train),"+-", dp_train)
print("test ", np.mean(y_test), "+-", dp_test)