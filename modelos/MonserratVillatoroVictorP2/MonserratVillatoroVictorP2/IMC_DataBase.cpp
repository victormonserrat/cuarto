#include "IMC_DataBase.hpp"

#include <string>
#include <fstream>
#include <vector>

imc::DataBase::DataBase(std::string fileName) {
  // Abrir fichero
	std::fstream file(fileName.c_str());
	// Leer número de entradas, salidas y patrones
	file >> _nNumEntradas >> _nNumSalidas >> _nNumPatrones;
	// Leer patrones
	double dAux;
	for (int i = 0; i < _nNumPatrones; i++) {
		// Leer entradas
		std::vector<double> e;
		for (int j = 0; j < _nNumEntradas; j++) {
			file >> dAux;
			e.push_back(dAux);
		}
		_entradas.push_back(e);
		// Leer salidas
		std::vector<double> s;
		for (int j = 0; j < _nNumSalidas; j++) {
			file >> dAux;
			s.push_back(dAux);
		}
		_salidas.push_back(s);
	}
}
