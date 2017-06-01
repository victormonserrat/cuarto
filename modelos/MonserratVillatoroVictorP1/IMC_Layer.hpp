#ifndef __IMC_LAYER_HPP__
#define __IMC_LAYER_HPP__

#include "IMC_Neuron.hpp"

#include <vector>

namespace imc {
  class Layer {
  public:
    int  nNumNeuronas() const {return _nNumNeuronas;}
    void nNumNeuronas(int const &n) {_nNumNeuronas = n;}
    std::vector<Neuron> &pNeuronas() {return _pNeuronas;}
    void pNeuronas(std::vector<Neuron> const &pNeuronas) {_pNeuronas = pNeuronas;}
  private:
    int                 _nNumNeuronas; /* Número de neuronas de la capa */
    std::vector<Neuron> _pNeuronas;    /* Vector con las neuronas de la capa */
  };
}

#endif
