#ifndef __IMC_DATABASE_HPP__
#define __IMC_DATABASE_HPP__

#include <string>
#include <vector>

namespace imc {
  class DataBase {
  public:
    DataBase(std::string fileName);
    int  nNumEntradas() const {return _nNumEntradas;}
    void nNumEntradas(int const &n) {_nNumEntradas = n;}
    int  nNumSalidas() const {return _nNumSalidas;}
    void nNumSalidas(int const &n) {_nNumSalidas = n;}
    int  nNumPatrones() const {return _nNumPatrones;}
    void nNumPatroes(int const &n) {_nNumPatrones = n;}
    std::vector<std::vector<double>> entradas() const {return _entradas;}
    std::vector<std::vector<double>> salidas() const {return _salidas;}
  private:
    int _nNumEntradas;                          /* Número de entradas */
    int _nNumSalidas;                           /* Número de salidas */
    int _nNumPatrones;                          /* Número de patrones */
    std::vector<std::vector<double>> _entradas; /* Matriz con las entradas del problema */
    std::vector<std::vector<double>> _salidas;  /* Matriz con las salidas del problema */
  };
}

#endif
