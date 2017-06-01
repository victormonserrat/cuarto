#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TODO: Incluir todos los import necesarios
"""
import sys
# Tiempo computacional
# import time

import pandas as pd
import numpy as np

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
# CCR sin clasifiación
# from sklearn.metrics import accuracy_score
# Matriz de confusión
# from sklearn.metrics import confusion_matrix

def entrenar_rbf(fichero_train, fichero_test, num_rbf, clasificacion, eta, l2):
    """ Función principal
        Recibe los siguientes parámetros:
            - fichero_train: nombre del fichero de entrenamiento.
            - fichero_test: nombre del fichero de test.
            - num_rbf: número de neuronas de tipo RBF.
            - clasificacion: True si el problema es de clasificacion.
            - eta: valor del parámetro de regularización para la Regresión
              Logística.
            - l2: True si queremos utilizar L2 para la Regresión Logística.
              False si queremos usar L1.
        Devuelve:
            - train_mse: Error de tipo Mean Squared Error en entrenamiento.
              En el caso de clasificación, calcularemos el MSE de las
              probabilidades predichas frente a las objetivo.
            - test_mse: Error de tipo Mean Squared Error en test.
              En el caso de clasificación, calcularemos el MSE de las
              probabilidades predichas frente a las objetivo.
            - train_ccr: Error de clasificación en entrenamiento.
              En el caso de regresión, devolvemos un cero.
            - test_ccr: Error de clasificación en test.
              En el caso de regresión, devolvemos un cero.
    """
    train_inputs, train_outputs, test_inputs, test_outputs = lectura_datos(fichero_train,
                                                                           fichero_test)

    kmedias, distancias, centros = clustering(clasificacion, train_inputs,
                                              train_outputs, num_rbf)

    radios = calcular_radios(centros, num_rbf)

    matriz_r = calcular_matriz_r(distancias, radios)

    if not clasificacion:
        coeficientes = invertir_matriz_regresion(matriz_r, train_outputs)
        # Coeficientes
        # print coeficientes
    else:
        logreg = logreg_clasificacion(matriz_r, train_outputs, eta, l2)

    """
    TODO: Calcular las distancias de los centroides a los patrones de test
          y la matriz R de test
    """
    distancias_test = kmedias.transform(test_inputs)
    matriz_r_test = calcular_matriz_r(distancias_test, radios)

    if not clasificacion:
        """
        TODO: Obtener las predicciones de entrenamiento y de test y calcular
              el MSE
        """
        prediccion = np.dot(matriz_r, coeficientes)
        prediccion_test = np.dot(matriz_r_test, coeficientes)
        train_ccr = 0
        test_ccr = 0
        # CCR sin clasifiación
        # prediccion = np.around(prediccion)
        # prediccion_test = np.around(prediccion_test)
        # train_ccr = accuracy_score(prediccion, train_outputs)*100
        # test_ccr = accuracy_score(prediccion_test, test_outputs)*100
    else:
        """
        TODO: Obtener las predicciones de entrenamiento y de test y calcular
              el CCR. Calcular también el MSE, comparando las probabilidades
              obtenidas y las probabilidades objetivo
        """
        prediccion = logreg.predict(matriz_r)
        prediccion_test = logreg.predict(matriz_r_test)
        train_ccr = logreg.score(matriz_r, train_outputs)*100
        test_ccr = logreg.score(matriz_r_test, test_outputs)*100

    train_mse = mean_squared_error(train_outputs, prediccion)
    test_mse = mean_squared_error(test_outputs, prediccion_test)
    # Matriz de confusión
    # print confusion_matrix(test_outputs, prediccion_test)

    return train_mse, test_mse, train_ccr, test_ccr

def lectura_datos(fichero_train, fichero_test):
    """ Realiza la lectura de datos.
        Recibe los siguientes parámetros:
            - fichero_train: nombre del fichero de entrenamiento.
            - fichero_test: nombre del fichero de test.
        Devuelve:
            - train_inputs: matriz con las variables de entrada de
              entrenamiento.
            - train_outputs: matriz con las variables de salida de
              entrenamiento.
            - test_inputs: matriz con las variables de entrada de
              test.
            - test_outputs: matriz con las variables de salida de
              test.
    """

    """
    TODO: Completar el código de la función
    """
    dataset = pd.read_csv(fichero_train, header=None)
    train_inputs = dataset.values[:, 0:-1]
    train_outputs = dataset.values[:, -1:]
    dataset = pd.read_csv(fichero_test, header=None)
    test_inputs = dataset.values[:, 0:-1]
    test_outputs = dataset.values[:, -1:]

    return train_inputs, train_outputs, test_inputs, test_outputs

def inicializar_centroides_clas(train_inputs, train_outputs, num_rbf):
    """ Inicializa los centroides para el caso de clasificación.
        Debe elegir, aprox., num_rbf/num_clases patrones por cada clase.
        Recibe los siguientes parámetros:
            - train_inputs: matriz con las variables de entrada de
              entrenamiento.
            - train_outputs: matriz con las variables de salida de
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - centroides: matriz con todos los centroides iniciales
                          (num_rbf x num_entradas).
    """

    """
    TODO: Completar el código de la función
    """
    sss = StratifiedShuffleSplit(n_splits=1, train_size=num_rbf, test_size=None)
    centroides_generator = sss.split(train_inputs, train_outputs)
    centroides_index = list(centroides_generator)[0][0]
    centroides = train_inputs[centroides_index]

    return centroides

def clustering(clasificacion, train_inputs, train_outputs, num_rbf):
    """ Realiza el proceso de clustering. En el caso de la clasificación, se
        deben escoger los centroides usando inicializar_centroides_clas()
        En el caso de la regresión, se escogen aleatoriamente.
        Recibe los siguientes parámetros:
            - clasificacion: True si el problema es de clasificacion.
            - train_inputs: matriz con las variables de entrada de
              entrenamiento.
            - train_outputs: matriz con las variables de salida de
              entrenamiento.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - kmedias: objeto de tipo sklearn.cluster.KMeans ya entrenado.
            - distancias: matriz (num_patrones x num_rbf) con la distancia
              desde cada patrón hasta cada rbf.
            - centros: matriz (num_rbf x num_entradas) con los centroides
              obtenidos tras el proceso de clustering.
    """

    """
    TODO: Completar el código de la función
    """
    if clasificacion: # Clasificación
        centros = inicializar_centroides_clas(train_inputs, train_outputs, num_rbf)
        kmedias = KMeans(n_clusters=num_rbf, init=centros, n_init=1, max_iter=500)
    else: # Regresión
        kmedias = KMeans(n_clusters=num_rbf, init='random', max_iter=500)

    distancias = kmedias.fit_transform(train_inputs)
    centros = kmedias.cluster_centers_

    return kmedias, distancias, centros

def calcular_radios(centros, num_rbf):
    """ Calcula el valor de los radios tras el clustering.
        Recibe los siguientes parámetros:
            - centros: conjunto de centroides.
            - num_rbf: número de neuronas de tipo RBF.
        Devuelve:
            - radios: vector (num_rbf) con el radio de cada RBF.
    """

    """
    TODO: Completar el código de la función
    """
    distancias = pdist(centros)
    sumas_distancias = squareform(distancias)
    sumas_distancias = sumas_distancias.sum(axis=1)
    radios = sumas_distancias/(2*(num_rbf-1))

    return radios

def calcular_matriz_r(distancias, radios):
    """ Devuelve el valor de activación de cada neurona para cada patrón
        (matriz R en la presentación)
        Recibe los siguientes parámetros:
            - distancias: matriz (num_patrones x num_rbf) con la distancia
              desde cada patrón hasta cada rbf.
            - radios: array (num_rbf) con el radio de cada RBF.
        Devuelve:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al final, en la última columna, un vector con todos los
              valores a 1, que actuará como sesgo.
    """

    """
    TODO: Completar el código de la función
    """
    distancias_cuadradas = np.square(distancias)
    radios_cuadrados = np.square(radios)
    matriz_r = np.exp(-distancias_cuadradas/(2*radios_cuadrados))
    sesgo = np.ones((distancias.shape[0], 1))
    matriz_r = np.append(matriz_r, sesgo, axis=1)

    return matriz_r

def invertir_matriz_regresion(matriz_r, train_outputs):
    """ Devuelve el vector de coeficientes obtenidos para el caso de la
        regresión (matriz beta en las diapositivas)
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al principio, en la última columna, un vector con todos los
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de
              entrenamiento.
        Devuelve:
            - coeficientes: vector (num_rbf+1) con el valor del sesgo y del
              coeficiente de salida para cada rbf.
    """

    """
    TODO: Completar el código de la función
    """
    num_patrones = matriz_r.shape[0]
    num_rbf = matriz_r.shape[1]-1

    if num_patrones == num_rbf+1: # La matriz es cuadrada
        inversa_r =  np.linalg.inv(matriz_r) # Inversa
    elif num_rbf+1 < num_patrones: # La matriz no es cuadrada y tiene solución única
        inversa_r = np.linalg.pinv(matriz_r) # Pseudoinversa
    else: # La matriz es cuadrada y existen muchas soluciones
        print "Existen muchas soluciones y hay que usar algún tipo de algoritmo de reducción de caracterı́sticas para bajar el valor de num_rbf."
        sys.exit(-1)

    coeficientes_traspuesta = np.dot(inversa_r, train_outputs)
    coeficientes = coeficientes_traspuesta.transpose()[0]

    return coeficientes

def logreg_clasificacion(matriz_r, train_outputs, eta, l2):
    """ Devuelve el objeto de tipo regresión logística obtenido a partir de la
        matriz R.
        Recibe los siguientes parámetros:
            - matriz_r: matriz (num_patrones x (num_rbf+1)) con el valor de
              activación (out) de cada RBF para cada patrón. Además, añadimos
              al principio, en la primera columna, un vector con todos los
              valores a 1, que actuará como sesgo.
            - train_outputs: matriz con las variables de salida de
              entrenamiento.
            - eta: valor del parámetro de regularización para la Regresión
              Logística.
            - l2: True si queremos utilizar L2 para la Regresión Logística.
              False si queremos usar L1.
        Devuelve:
            - logreg: objeto de tipo sklearn.linear_model.ssion ya
              entrenado.
    """

    """
    TODO: Completar el código de la función
    """
    if l2:
        l = 'l2'
    else:
        l = 'l1'

    if np.isclose(eta, 0):
        c = 10e-15
    else:
        c = 1/eta
    logreg = LogisticRegression(penalty=l, C=c, fit_intercept=False).fit(matriz_r, train_outputs[:, 0])

    return logreg

if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(-1)
    pyName, train, test, num_rbf, clasificacion, eta, l2 = sys.argv
    num_rbf = int(num_rbf)
    if clasificacion == "0":
        clasificacion = False
    else:
        clasificacion = True
    eta = float(eta)
    if l2 == "0":
        l2 = False
    else:
        l2 = True

    train_mses = np.empty(5)
    train_ccrs = np.empty(5)
    test_mses = np.empty(5)
    test_ccrs = np.empty(5)
    for s in range(10,60,10):
        print "-----------"
        print "Semilla: %d" % s
        print "-----------"

        np.random.seed(s)
        # Tiempo computacional
        # start_time = time.time()
        train_mses[s/10-1], test_mses[s/10-1], train_ccrs[s/10-1], test_ccrs[s/10-1] = \
            entrenar_rbf(train, test, num_rbf, clasificacion, eta, l2)
        # Tiempo computacional
        # elapsed_time = time.time() - start_time
        # print elapsed_time
        print "MSE de entrenamiento: %f" % train_mses[s/10-1]
        print "MSE de test: %f" % test_mses[s/10-1]
        if clasificacion:
            print "CCR de entrenamiento: %.2f%%" % train_ccrs[s/10-1]
            print "CCR de test: %.2f%%" % test_ccrs[s/10-1]
        # CCR sin clasifiación
        # print "CCR de entrenamiento: %.2f%%" % train_ccrs[s / 10 - 1]
        # print "CCR de test: %.2f%%" % test_ccrs[s / 10 - 1]

    """
    TODO: Imprimir la media y la desviación típica del MSE y del CCR
    """
    print "*********************"
    print "Resumen de resultados"
    print "*********************"
    print "MSE de entrenamiento: %f +- %f" %(np.mean(train_mses), np.std(train_mses))
    print "MSE de test: %f +- %f" %(np.mean(test_mses), np.std(test_mses))
    if clasificacion:
        print "CCR de entrenamiento: %f +- %f" %(np.mean(train_ccrs), np.std(train_ccrs))
        print "CCR de test: %f +- %f" %(np.mean(test_ccrs), np.std(test_ccrs))
    # CCR sin clasifiación
    # print "CCR de entrenamiento: %.2f%%" % train_ccrs[s / 10 - 1]
    # print "CCR de test: %.2f%%" % test_ccrs[s / 10 - 1]
