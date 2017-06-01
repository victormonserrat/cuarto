# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# Cargar los datasets
X_train_sparse, y_train = load_svmlight_file("train_spam.libsvm")
X_test_sparse, y_test = load_svmlight_file("test_spam.libsvm")

# Convertirlos a formato denso
X_train = np.array(X_train_sparse.todense())
X_test = np.array(X_test_sparse.todense())

# Partición estratificada
#sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2)
#train_index, test_index = list(sss.split(X, y))[0]
#X_train = X[train_index]
#y_train = y[train_index]
#X_test = X[test_index]
#y_test = y[test_index]

# Estandarización
#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Configuración del modelo SVM
svm_model = svm.SVC(kernel='rbf')

# C ∈ {0.01, 0.1, 1, 10} y γ ∈ {0.01, 0.1, 1, 10}
Cs = np.logspace(-2, 1, num=4, base=10)
Gs = Cs

# Validación cruzada anidada tipo K-fold
optimo = GridSearchCV(estimator=svm_model, param_grid=dict(C=Cs, gamma=Gs), n_jobs=-1, cv=5)

# Tiempo computacional
start_time = time.time()

# Entrenar el modelo óptimo
optimo.fit(X_train, y_train)

# Tiempo computacional
elapsed_time = time.time() - start_time
print elapsed_time

# Configuración del modelo óptimo
print optimo.best_params_

# CCR de test óptimo
print optimo.score(X_test, y_test)*100

# Matriz de confusión
prediccion_test = optimo.predict(X_test)
print confusion_matrix(y_test, prediccion_test)

# Patrones mal clasificados
print X_test[prediccion_test != y_test]

# Representar los puntos
#plt.figure(1)
#plt.clf()
#plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

# Representación gráfica de la SVM
# --------------------------------
#plt.axis('tight')

# Extraer límites
#x_min = X[:, 0].min()
#x_max = X[:, 0].max()
#y_min = X[:, 1].min()
#y_max = X[:, 1].max()

# Crear un grid con todos los puntos y obtener el valor Z devuelto por la SVM
#XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
#Z = optimo.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Hacer un plot a color con los resultados
#Z = Z.reshape(XX.shape)
#plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
#plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                #levels=[-.5, 0, .5])

#plt.show()
