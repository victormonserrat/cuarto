#include "IMC_MultilayerPerceptron.hpp"
#include "IMC_Layer.hpp"
#include "IMC_Neuron.hpp"
#include "IMC_DataBase.hpp"

#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>

#define MAX_I 999
#define MAX_NO_IMPROVEMENT 50
#define TOLERANCE 0.00001

// Obtener un número real aleatorio en el intervalo [Low,High]
double imc::MultilayerPerceptron::realAleatorio(double const &Low, double const &High)
{
	return Low + ((double)rand() / RAND_MAX) * (High - Low);
}
// nl tiene el numero de capas y npl es un vector que contiene el número de neuronas por cada una de las capas
// Rellenar vector pCapas
void imc::MultilayerPerceptron::inicializar(int const &nl, std::vector<int> const &npl) {
	imc::Layer l;
	l.pNeuronas(std::vector<imc::Neuron>(npl[0]));
	l.nNumNeuronas(npl[0]);
	_pCapas.push_back(l);
	int s = 0;
	if (_bSesgo) {
		s = 1;
	}
	for (int i = 1; i < nl; i++) {
		imc::Layer l;
		for (int j = 0; j < npl[i]; j++) {
			imc::Neuron n;
			n.w(std::vector<double>(npl[i-1]+s));
			n.deltaW(std::vector<double>(npl[i-1]+s));
			n.ultimoDeltaW(std::vector<double>(npl[i-1]+s));
			n.wCopia(std::vector<double>(npl[i-1]+s));
			l.pNeuronas().push_back(n);
		}
		l.nNumNeuronas(npl[i]);
		_pCapas.push_back(l);
	}
	_nNumCapas = nl;
}
// Rellenar todos los pesos (w) aleatoriamente entre -1 y 1
void imc::MultilayerPerceptron::pesosAleatorios() {
	for (int i = 1; i < _nNumCapas; i++) {
		for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
			for (int k = 0; k < _pCapas[i-1].nNumNeuronas(); k++) {
				_pCapas[i].pNeuronas()[j].w()[k] = realAleatorio(-1, 1);
			}
			if (_bSesgo) {
				_pCapas[i].pNeuronas()[j].w()[_pCapas[i-1].nNumNeuronas()] = realAleatorio(-1, 1);
			}
		}
	}
}
// Alimentar las neuronas de entrada de la red con un patrón pasado como argumento
void imc::MultilayerPerceptron::alimentarEntradas(std::vector<double> const &input) {
	for (int i = 0; i < _pCapas[0].nNumNeuronas(); i++) {
		_pCapas[0].pNeuronas()[i].x(input[i]);
	}
}
// Recoger los valores predichos por la red (out de la capa de salida) y almacenarlos en el vector pasado como argumento
void imc::MultilayerPerceptron::recogerSalidas(std::vector<double> &output) {
	for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
		output[i] = _pCapas[_nNumCapas-1].pNeuronas()[i].x();
	}
}
// Hacer una copia de todos los pesos (copiar w en copiaW)
void imc::MultilayerPerceptron::copiarPesos() {
	for (int i = 1; i < _nNumCapas; i++) {
		for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
			_pCapas[i].pNeuronas()[j].wCopia(_pCapas[i].pNeuronas()[j].w());
		}
	}
}
// Restaurar una copia de todos los pesos (copiar copiaW en w)
void imc::MultilayerPerceptron::restaurarPesos() {
	for (int i = 1; i < _nNumCapas; i++) {
		for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
			_pCapas[i].pNeuronas()[j].w(_pCapas[i].pNeuronas()[j].wCopia());
		}
	}
}
// Calcular y propagar las salidas de las neuronas, desde la primera capa hasta la última
void imc::MultilayerPerceptron::propagarEntradas() {
	for (int i = 1; i < _nNumCapas; i++) {
		for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
			double net = 0.0;
			for (int k = 0; k < _pCapas[i-1].nNumNeuronas(); k++) {
				net += _pCapas[i].pNeuronas()[j].w()[k] * _pCapas[i-1].pNeuronas()[k].x();
			}
			if (_bSesgo) {
				net += _pCapas[i].pNeuronas()[j].w()[_pCapas[i-1].nNumNeuronas()];
			}
			_pCapas[i].pNeuronas()[j].x(1/(1+exp(-net)));
		}
	}
}
// Calcular el error de salida (MSE) del out de la capa de salida con respecto a un vector objetivo y devolverlo
double imc::MultilayerPerceptron::calcularErrorSalida(std::vector<double> const &target) {
	double error = 0.0;
	for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
		error += pow(_pCapas[_nNumCapas-1].pNeuronas()[i].x() - target[i], 2);
	}
	return error / _pCapas[_nNumCapas-1].nNumNeuronas();
}
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
void imc::MultilayerPerceptron::retropropagarError(std::vector<double> const &objetivo) {
	for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
		double out = _pCapas[_nNumCapas-1].pNeuronas()[i].x();
		_pCapas[_nNumCapas-1].pNeuronas()[i].dX(-1 * (objetivo[i] - out) * out * (1-out));
	}
	for (int i = _nNumCapas-2; i >= 0; i--) {
		for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
			double s = 0.0;
			for (int k = 0; k < _pCapas[i+1].nNumNeuronas(); k++) {
				s += _pCapas[i+1].pNeuronas()[k].w()[j] * _pCapas[i+1].pNeuronas()[k].dX();
			}
			double out = _pCapas[i].pNeuronas()[j].x();
			_pCapas[i].pNeuronas()[j].dX(s * out * (1-out));
		}
	}
}
// Acumular los cambios producidos por un patrón en deltaW
void imc::MultilayerPerceptron::acumularCambio() {
	for (int i = 1; i < _nNumCapas; i++) {
		for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
			for (int k = 0; k < _pCapas[i-1].nNumNeuronas(); k++) {
				_pCapas[i].pNeuronas()[j].deltaW()[k] += _pCapas[i].pNeuronas()[j].dX() * _pCapas[i-1].pNeuronas()[k].x();
			}
			if (_bSesgo) {
				_pCapas[i].pNeuronas()[j].deltaW()[_pCapas[i-1].nNumNeuronas()] += _pCapas[i].pNeuronas()[j].dX();
			}
		}
	}
}
// Actualizar los pesos de la red, desde la primera capa hasta la última
void imc::MultilayerPerceptron::ajustarPesos() {
	for (int i = 1; i < _nNumCapas; i++) {
		for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
			for (int k = 0; k < _pCapas[i-1].nNumNeuronas(); k++) {
				_pCapas[i].pNeuronas()[j].w()[k] += -_dEta * _pCapas[i].pNeuronas()[j].deltaW()[k] - _dMu * (_dEta * _pCapas[i].pNeuronas()[j].ultimoDeltaW()[k]);
			}
			if (_bSesgo) {
				_pCapas[i].pNeuronas()[j].w()[_pCapas[i-1].nNumNeuronas()] += -_dEta * _pCapas[i].pNeuronas()[j].deltaW()[_pCapas[i-1].nNumNeuronas()] - _dMu * (_dEta * _pCapas[i].pNeuronas()[j].ultimoDeltaW()[_pCapas[i-1].nNumNeuronas()]);
			}
		}
	}
}
// Imprimir la red, es decir, todas las matrices de pesos
void imc::MultilayerPerceptron::imprimirRed() {
	for (int i = 1; i < _nNumCapas; i++) {
		std::cout << "Capa " << i << std::endl;
		std::cout << "------" << std::endl;
		for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
			if (_bSesgo) {
				std::cout << _pCapas[i].pNeuronas()[j].w()[_pCapas[i-1].nNumNeuronas()] << " ";
			}
			for (int k = 0; k < _pCapas[i-1].nNumNeuronas(); k++) {
				std::cout << _pCapas[i].pNeuronas()[j].w()[k] << " ";
			}
			std::cout << std::endl;
		}
	}
}
// Simular la red: propagar las entradas hacia delante, computar el error, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón y objetivo es el vector de salidas deseadas del patrón
void imc::MultilayerPerceptron::simularRedOnline(std::vector<double> const &entrada, std::vector<double> const &objetivo) {
	for (int i = 1; i < _nNumCapas; i++) {
		for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
			for (int k = 0; k < _pCapas[i-1].nNumNeuronas(); k++) {
				_pCapas[i].pNeuronas()[j].ultimoDeltaW()[k] = _pCapas[i].pNeuronas()[j].deltaW()[k];
				_pCapas[i].pNeuronas()[j].deltaW()[k] = 0;
			}
		}
	}
	alimentarEntradas(entrada);
	propagarEntradas();
	retropropagarError(objetivo);
	acumularCambio();
	ajustarPesos();
}
// Entrenar la red on-line para un determinado fichero de datos
void imc::MultilayerPerceptron::entrenarOnline(DataBase const &pDatosTrain) {
	for (int i = 0; i < pDatosTrain.nNumPatrones(); i++) {
		simularRedOnline(pDatosTrain.entradas()[i], pDatosTrain.salidas()[i]);
	}
}
// Probar la red con un conjunto de datos y devolver el error MSE cometido
double imc::MultilayerPerceptron::test(DataBase const &pDatosTest) {
	double dAvgTestError = 0.0;
	for (int i = 0; i < pDatosTest.nNumPatrones(); i++) {
		alimentarEntradas(pDatosTest.entradas()[i]);
		propagarEntradas();
		dAvgTestError += calcularErrorSalida(pDatosTest.salidas()[i]);
	}
	dAvgTestError /= pDatosTest.nNumPatrones();
	return dAvgTestError;
}
// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
// Una vez terminado, probar como funciona la red en pDatosTest
// Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
void imc::MultilayerPerceptron::ejecutarAlgoritmoOnline(DataBase const &pDatosTrain, DataBase const &pDatosTest, int const &maxiter, double &errorTrain, double &errorTest) {
	// Inicialización de pesos
	pesosAleatorios();
	// Aprendizaje del algoritmo
	double minTrainError = 0;
	int numSinMejorar;
	int countTrain = 0;
	do {
		entrenarOnline(pDatosTrain);
		double trainError = test(pDatosTrain);
		// El 0.00001 es un valor de tolerancia, podría parametrizarse
		if (countTrain == 0 || fabs(trainError - minTrainError) > TOLERANCE) {
			minTrainError = trainError;
			copiarPesos();
			numSinMejorar = 0;
		} else {
			numSinMejorar++;
		}
		if(numSinMejorar == MAX_NO_IMPROVEMENT){
			restaurarPesos();
			countTrain = maxiter-1;
		}
		countTrain++;
		std::cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError << std::endl;

	} while (countTrain < maxiter || countTrain == MAX_I);
	std::cout << "PESOS DE LA RED" << std::endl;
	std::cout << "===============" << std::endl;
	imprimirRed();
	std::cout << "Salida Esperada Vs Salida Obtenida (test)" << std::endl;
	std::cout << "=========================================" << std::endl;
	for (int i = 0; i < pDatosTest.nNumPatrones(); i++) {
		std::vector<double> prediccion(pDatosTest.nNumSalidas());
		// Cargamos las entradas y propagamos el valor
		alimentarEntradas(pDatosTest.entradas()[i]);
		propagarEntradas();
		recogerSalidas(prediccion);
		for (int j = 0; j < pDatosTest.nNumSalidas(); j++) {
			std::cout << pDatosTest.salidas()[i][j] << " -- " << prediccion[j];
		}
		std::cout << std::endl;
	}
	errorTest = test(pDatosTest);
	errorTrain = minTrainError;
}
