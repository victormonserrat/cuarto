#ifndef __IMC_NEURON_HPP__
#define __IMC_NEURON_HPP__

#include <vector>

namespace imc {
  class Neuron {
  public:
    double x() const {return _x;}
    void x(double const &x) {_x = x;}
    double dX() const {return _dX;}
    void dX(double const &dX) {_dX = dX;}
    std::vector<double> &w() {return _w;}
    void w(std::vector<double> const &w) {_w = w;}
    std::vector<double> &deltaW() {return _deltaW;}
    void deltaW(std::vector<double> const &deltaW) {_deltaW = deltaW;}
    std::vector<double> &ultimoDeltaW() {return _ultimoDeltaW;}
    void ultimoDeltaW(std::vector<double> const &ultimoDeltaW) {_ultimoDeltaW = ultimoDeltaW;}
    std::vector<double> &wCopia() {return _wCopia;}
    void wCopia(std::vector<double> const &wCopia) {_wCopia = wCopia;}
  private:
    double _x;            /* Salida producida por la neurona (out_j^h) */
    double _dX;            /* Derivada de la salida producida por la neurona (delta_j) */
    std::vector<double> _w;            /* Vector de pesos de entrada (w_{ji}^h) */
    std::vector<double> _deltaW;       /* Cambio a aplicar a cada peso de entrada (\Delta_{ji}^h (t)) */
    std::vector<double> _ultimoDeltaW; /* Último cambio aplicada a cada peso (\Delta_{ji}^h (t-1)) */
    std::vector<double> _wCopia;      /* Copia de los pesos de entrada */
  };
}

#endif
