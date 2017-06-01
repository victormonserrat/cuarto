# -*- coding: utf-8 -*-

import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# Número de clases
n_classes = 7

# Cargar el dataset
X_sparse, y = load_svmlight_file("glass.libsvm")

# Convertirlo a formato denso
X = np.array(X_sparse.todense())

# Generación de centroides
centroids = np.empty([n_classes, X.shape[1]])
for c in range(n_classes):
    centroids[c] = np.mean(X[y == c], axis=0)

# Medidas de evaluación
mse = 0
ccr = 0

# Validación cruzada (k=10)
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

   # k-Means
   kmeans = KMeans(n_clusters=n_classes, n_init=1, init=centroids).fit(X_train)
   prediction = kmeans.predict(X_test)

   # Evaluación
   mse += mean_squared_error(y_test, prediction)
   ccr += accuracy_score(prediction, y_test)*100

# Evaluación Medidas
mse /= 10
print mse
ccr /= 10
print ccr
