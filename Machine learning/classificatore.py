import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#dati che verrano utilizzati
iris_dataset = datasets.load_iris()
print(iris_dataset["DESCR"])

#caratteristiche, dati in input
x = iris_dataset.data

#output, cioe' quello che il modello dovrebbe predire
y = iris_dataset.target

#divido i dati, un parte li uso per addestrare, l'altra per
#testare se il modello ha imparato bene
x_train, x_test, y_train, y_test = train_test_split(x, y)

#modello non addestrato, classificatore
#un classificatore prende i dati e restituisce un categoria
modello = DecisionTreeClassifier()

#addestro il modello
modello.fit(x_train, y_train)

#predizioni sui dati su cui ha imparato
predizione_train = modello.predict(x_train)

#predizioni su nuovoi dati
predizione_test = modello.predict(x_test)

#misuro l'accuratezza sia dell'addestramento che del test
#questo puo' dare informazioni su over fitting o meno
#sinceramente non so come
print("accuratezza train")
print(accuracy_score(y_train, predizione_train))

print("accuratezza test")
print(accuracy_score(y_test, predizione_test))

#Rappresentazione grafica dei quanto e' stato bravo il modello
#sulle y c'e' la risposta ceh il modello doveva dare e
#sulle x ci sta la predizione che il modello ha dato, quindi gli
#elementi fuori diagonali sono le risposte sbagliate
skplt.metrics.plot_confusion_matrix(y_train,predizione_train)
skplt.metrics.plot_confusion_matrix(y_test,predizione_test)

plt.show()