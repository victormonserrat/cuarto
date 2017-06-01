#ifndef __IMC_STATIST_HPP__
#define __IMC_STATIST_HPP__

#include <cmath>

namespace imc {
  /* Calcular la media */
  double mean(std::vector<double> const &x) {
    double s = 0;
    for (size_t i = 0; i < x.size(); i++) {
      s += x[i];
    }
    return s/x.size();
  }
  /* Calcular la desviación típica */
  double sd(std::vector<double> const &x, double const &m) {
    double s = 0;
    for (size_t i = 0; i < x.size(); i++) {
      s += pow(x[i]-m, 2);
    }
    return sqrt(s/x.size());
  }
}

#endif
