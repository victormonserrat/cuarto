#include "IMC_MultilayerPerceptron.hpp"
#include "IMC_Layer.hpp"
#include "IMC_Neuron.hpp"
#include "IMC_DataBase.hpp"

#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>

#define MAX_I 1000
#define MAX_NO_IMPROVEMENT 50
#define TOLERANCE 0.00001

// Obtener un número entero aleatorio en el intervalo [Low,High]
int imc::MultilayerPerceptron::enteroAleatorio(int const &Low, int const &High) {
	return rand() % (High - Low + 1) + Low;
}
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
	for (int i = 1; i < nl; i++) {
		imc::Layer l;
		for (int j = 0; j < npl[i]; j++) {
			imc::Neuron n;
			n.w(std::vector<double>(npl[i-1]+_bSesgo));
			n.deltaW(std::vector<double>(npl[i-1]+_bSesgo));
			n.ultimoDeltaW(std::vector<double>(npl[i-1]+_bSesgo, 0));
			n.wCopia(std::vector<double>(npl[i-1]+_bSesgo));
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
			for (int k = 0; k < _pCapas[i-1].nNumNeuronas()+_bSesgo; k++) {
				_pCapas[i].pNeuronas()[j].w()[k] = realAleatorio(-1, 1);
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
	if (_bSigmoideCapaSalida) {
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
	} else {
		for (int i = 1; i < _nNumCapas-1; i++) {
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
		double sumNet = 0.0;
		std::vector<double> exponenciales(_pCapas[_nNumCapas-1].nNumNeuronas());
		for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
			double net = 0.0;
			for (int j = 0; j < _pCapas[_nNumCapas-2].nNumNeuronas(); j++) {
				net += _pCapas[_nNumCapas-1].pNeuronas()[i].w()[j] * _pCapas[_nNumCapas-2].pNeuronas()[j].x();
			}
			if (_bSesgo) {
				net += _pCapas[_nNumCapas-1].pNeuronas()[i].w()[_pCapas[_nNumCapas-2].nNumNeuronas()];
			}
			exponenciales[i] = exp(net);
			sumNet += exponenciales[i];
		}
		for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
			_pCapas[_nNumCapas-1].pNeuronas()[i].x(exponenciales[i]/sumNet);
		}
	}
}
// Calcular el error de salida (MSE) del out de la capa de salida con respecto a un vector objetivo y devolverlo
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double imc::MultilayerPerceptron::calcularErrorSalida(std::vector<double> const &target, int const &funcionError) {
	double error = 0.0;
	switch (funcionError) {
		case 0: {
			// MSE
			for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
				error += pow(_pCapas[_nNumCapas-1].pNeuronas()[i].x() - target[i], 2);
			}
			break;
		}
		case 1: {
			// EntropiaCruzada
			for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
				error += -target[i] * log(_pCapas[_nNumCapas-1].pNeuronas()[i].x());
			}
			break;
		}
	}
	error /= _pCapas[_nNumCapas-1].nNumNeuronas();
	return error;
}
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void imc::MultilayerPerceptron::retropropagarError(std::vector<double> const &objetivo, int const &funcionError) {
	// Sigmoide
	if (_bSigmoideCapaSalida) {
		switch (funcionError) {
			// MSE
			case 0:
				for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
					double outi = _pCapas[_nNumCapas-1].pNeuronas()[i].x();
					_pCapas[_nNumCapas-1].pNeuronas()[i].dX(-(objetivo[i] - outi) * outi * (1-outi));
				}
				break;
			// EntropiaCruzada
			case 1:
				for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
					double outi = _pCapas[_nNumCapas-1].pNeuronas()[i].x();
					_pCapas[_nNumCapas-1].pNeuronas()[i].dX(-(objetivo[i] / outi) * outi * (1-outi));
				}
				break;
		}
	// Softmax
	} else {
		switch (funcionError) {
			// MSE
			case 0: {
				for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
					double outi = _pCapas[_nNumCapas-1].pNeuronas()[i].x();
					double sum = 0.0;
					for (int j = 0; j < _pCapas[_nNumCapas-1].nNumNeuronas(); j++) {
						double outj = _pCapas[_nNumCapas-1].pNeuronas()[j].x();
						int cond;
						if (i == j) {
							cond = 1;
						} else {
							cond = 0;
						}
						sum += (objetivo[j] - outj) * outi * (cond - outj);
					}
					_pCapas[_nNumCapas-1].pNeuronas()[i].dX(-sum);
				}
				break;
			}
			// EntropiaCruzada
			case 1: {
				for (int i = 0; i < _pCapas[_nNumCapas-1].nNumNeuronas(); i++) {
					double outi = _pCapas[_nNumCapas-1].pNeuronas()[i].x();
					double sum = 0.0;
					for (int j = 0; j < _pCapas[_nNumCapas-1].nNumNeuronas(); j++) {
						double outj = _pCapas[_nNumCapas-1].pNeuronas()[j].x();
						int cond;
						if (i == j) {
							cond = 1;
						} else {
							cond = 0;
						}
						sum += (objetivo[j] / outj) * outi * (cond - outj);
					}
					_pCapas[_nNumCapas-1].pNeuronas()[i].dX(-sum);
				}
				break;
			}
		}
	}
	for (int i = _nNumCapas-2; i > 0; i--) {
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
			for (int k = 0; k < _pCapas[i-1].nNumNeuronas()+_bSesgo; k++) {
				_pCapas[i].pNeuronas()[j].w()[k] += -_dEta * _pCapas[i].pNeuronas()[j].deltaW()[k] - _dMu * (_dEta * _pCapas[i].pNeuronas()[j].ultimoDeltaW()[k]);
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
// Simular la red: propagar las entradas hacia delante, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón y objetivo es el vector de salidas deseadas del patrón
// El paso de ajustar pesos solo deberá hacerse si el algoritmo es on-line
// Si no lo es, el ajuste de pesos hay que hacerlo en la función "entrenar"
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void imc::MultilayerPerceptron::simularRed(std::vector<double> const &entrada, std::vector<double> const &objetivo, int const &funcionError) {
	if (_bOnLine) {
		for (int i = 1; i < _nNumCapas; i++) {
			for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
				for (int k = 0; k < _pCapas[i-1].nNumNeuronas(); k++) {
					_pCapas[i].pNeuronas()[j].ultimoDeltaW()[k] = _pCapas[i].pNeuronas()[j].deltaW()[k];
					_pCapas[i].pNeuronas()[j].deltaW()[k] = 0;
				}
			}
		}
	}
	alimentarEntradas(entrada);
	propagarEntradas();
	retropropagarError(objetivo, funcionError);
	acumularCambio();
	if (_bOnLine) {
		ajustarPesos();
	}
}
// Entrenar la red on-line para un determinado fichero de datos (pasar una vez por todos los patrones)
// Si es offline, después de pasar por ellos hay que ajustar pesos. Sino, ya se ha ajustado en cada patrón
void imc::MultilayerPerceptron::entrenar(DataBase const &pDatosTrain, int const &funcionError) {
	if (!_bOnLine) {
		for (int i = 1; i < _nNumCapas; i++) {
			for (int j = 0; j < _pCapas[i].nNumNeuronas(); j++) {
				for (int k = 0; k < _pCapas[i-1].nNumNeuronas(); k++) {
					_pCapas[i].pNeuronas()[j].ultimoDeltaW()[k] = _pCapas[i].pNeuronas()[j].deltaW()[k];
					_pCapas[i].pNeuronas()[j].deltaW()[k] = 0;
				}
			}
		}
	}
	for (int i = 0; i < pDatosTrain.nNumPatrones(); i++) {
		simularRed(pDatosTrain.entradas()[i], pDatosTrain.salidas()[i], funcionError);
	}
	if (!_bOnLine) {
		ajustarPesos();
	}
}
// Probar la red con un conjunto de datos y devolver el error MSE cometido
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double imc::MultilayerPerceptron::test(DataBase const &pDatosTest, int const &funcionError) {
	double dAvgTestError = 0.0;
	for (int i = 0; i < pDatosTest.nNumPatrones(); i++) {
		alimentarEntradas(pDatosTest.entradas()[i]);
		propagarEntradas();
		dAvgTestError += calcularErrorSalida(pDatosTest.salidas()[i], funcionError);
	}
	dAvgTestError /= pDatosTest.nNumPatrones();
	return dAvgTestError;
}
// Probar la red con un conjunto de datos y devolver el error CCR cometido
double imc::MultilayerPerceptron::testClassification(DataBase const &pDatosTest) {
	double ccr = 0.0;
	for (int i = 0; i < pDatosTest.nNumPatrones(); i++) {
		alimentarEntradas(pDatosTest.entradas()[i]);
		propagarEntradas();
		int dp = 0, op = 0;
		double ydp = 0.0, yop = 0.0;
		for (int j = 0; j < pDatosTest.nNumSalidas(); j++) {
			// ArgMax dp
			if (pDatosTest.salidas()[i][j] > ydp) {
				ydp = pDatosTest.salidas()[i][j];
				dp = j;
			}
			// ArgMax op
			if (_pCapas[_nNumCapas-1].pNeuronas()[j].x() > yop) {
				yop = _pCapas[_nNumCapas-1].pNeuronas()[j].x();
				op = j;
			}
		}
		if (dp == op) {
			ccr++;
		}
	}
	ccr /= pDatosTest.nNumPatrones();
	ccr *= 100;
	return ccr;
}
// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
// Una vez terminado, probar como funciona la red en pDatosTest
// Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void imc::MultilayerPerceptron::ejecutarAlgoritmo(DataBase const &pDatosTrain, DataBase const &pDatosTest, int const &maxiter, double &errorTrain, double &errorTest, double &ccrTrain, double &ccrTest, int const &funcionError) {
	// Inicialización de pesos
	pesosAleatorios();
	// Aprendizaje del algoritmo
	double minTrainError = 0;
	int numSinMejorar;
	int countTrain = 0;
	do {
		entrenar(pDatosTrain, funcionError);
		double trainError = test(pDatosTrain, funcionError);
		// El 0.00001 es un valor de tolerancia, podría parametrizarse
		if (countTrain == 0 || fabs(trainError - minTrainError) > TOLERANCE) {
			minTrainError = trainError;
			copiarPesos();
			numSinMejorar = 0;
		} else {
			numSinMejorar++;
		}
		if(numSinMejorar == MAX_NO_IMPROVEMENT){
			countTrain = maxiter-1;
		}
		countTrain++;
		std::cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError << std::endl;

	} while (countTrain < maxiter && countTrain < MAX_I);
	restaurarPesos();
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
			std::cout << pDatosTest.salidas()[i][j] << " -- " << prediccion[j] << " \\ ";
		}
		std::cout << std::endl;
	}
	errorTest = test(pDatosTest, funcionError);
	errorTrain = minTrainError;
	ccrTest = testClassification(pDatosTest);
	ccrTrain = testClassification(pDatosTrain);
}
