# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_svmlight_file
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# Cargar el dataset
X_sparse, y = load_svmlight_file("dataset3.libsvm")

# Convertirlo a formato denso
X = np.array(X_sparse.todense())

# Partición estratificada
sss = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2)
train_index, test_index = list(sss.split(X, y))[0]
X_train = X[train_index]
y_train = y[train_index]
X_test = X[test_index]
y_test = y[test_index]

# Estandarización
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Configuración del modelo SVM
svm_model = svm.SVC(kernel='rbf')

# C ∈ {0.02, 0.2, 2, 200} y γ ∈ {0.02, 0.2, 2, 200}
Cs = 2*np.logspace(-2, 0, num=3, base=10)
Cs = np.append(Cs, 200)
Gs = Cs

# Validación cruzada anidada tipo K-fold
optimo = GridSearchCV(estimator=svm_model, param_grid=dict(C=Cs, gamma=Gs), n_jobs=-1, cv=5)

# Entrenar el modelo óptimo
optimo.fit(X_train, y_train)

# Configuración del modelo óptimo
print optimo.best_params_

# CCR de test óptimo
print optimo.score(X_test, y_test)*100

# Representar los puntos
plt.figure(1)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

# Representación gráfica de la SVM
# --------------------------------
plt.axis('tight')

# Extraer límites
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()

# Crear un grid con todos los puntos y obtener el valor Z devuelto por la SVM
XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
Z = optimo.decision_function(np.c_[XX.ravel(), YY.ravel()])

# Hacer un plot a color con los resultados
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

plt.show()
