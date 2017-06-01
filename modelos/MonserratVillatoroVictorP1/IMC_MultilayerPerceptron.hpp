#ifndef __MULTILAYER_PERCEPTRON_HPP__
#define __MULTILAYER_PERCEPTRON_HPP__

#include "IMC_Layer.hpp"
#include "IMC_DataBase.hpp"

#include <vector>

namespace imc {
  class MultilayerPerceptron {
  public:
    int nNumCapas() const {return _nNumCapas;}
    void nNumCapas(int const &n) {_nNumCapas = n;}
    std::vector<Layer> &pCapas() {return _pCapas;}
    void pCapas(std::vector<Layer> const &l) {_pCapas = l;}
    double dEta() const {return _dEta;}
    void dEta(double const &e) {_dEta = e;}
    double dMu() const {return _dMu;}
    void dMu(double const &m) {_dMu = m;}
    bool bSesgo() const {return _bSesgo;}
    void bSesgo(bool const &b) {_bSesgo = b;}

    // Obtener un número real aleatorio en el intervalo [Low,High]
    double realAleatorio(double const &Low, double const &High);
    // nl tiene el numero de capas y npl es un vector que contiene el número de neuronas por cada una de las capas
    // Rellenar vector pCapas
    void inicializar(int const &nl, std::vector<int> const &npl);
    // Liberar memoria para las estructuras de datos
    void liberarMemoria();
    // Rellenar todos los pesos (w) aleatoriamente entre -1 y 1
    void pesosAleatorios();
    // Alimentar las neuronas de entrada de la red con un patrón pasado como argumento
    void alimentarEntradas(std::vector<double> const &input);
    // Recoger los valores predichos por la red (out de la capa de salida) y almacenarlos en el vector pasado como argumento
    void recogerSalidas(std::vector<double> &output);
    // Hacer una copia de todos los pesos (copiar w en copiaW)
    void copiarPesos();
    // Restaurar una copia de todos los pesos (copiar copiaW en w)
    void restaurarPesos();
    // Calcular y propagar las salidas de las neuronas, desde la primera capa hasta la última
    void propagarEntradas();
    // Calcular el error de salida (MSE) del out de la capa de salida con respecto a un vector objetivo y devolverlo
    double calcularErrorSalida(std::vector<double> const &target);
    // Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
    void retropropagarError(std::vector<double> const &objetivo);
    // Acumular los cambios producidos por un patrón en deltaW
    void acumularCambio();
    // Actualizar los pesos de la red, desde la primera capa hasta la última
    void ajustarPesos();
    // Imprimir la red, es decir, todas las matrices de pesos
    void imprimirRed();
    // Simular la red: propagar las entradas hacia delante, computar el error, retropropagar el error y ajustar los pesos
    // entrada es el vector de entradas del patrón y objetivo es el vector de salidas deseadas del patrón
    void simularRedOnline(std::vector<double> const &entrada, std::vector<double> const &objetivo);
    // Entrenar la red on-line para un determinado fichero de datos
    void entrenarOnline(DataBase const &pDatosTrain);
    // Probar la red con un conjunto de datos y devolver el error MSE cometido
    double test(DataBase const &pDatosTest);
    // Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
    // Una vez terminado, probar como funciona la red en pDatosTest
    // Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
    void ejecutarAlgoritmoOnline(DataBase const &pDatosTrain, DataBase const &pDatosTest, int const &maxiter, double &errorTrain, double &errorTest);

  private:
    int                _nNumCapas; /* Número de capas total en la red */
    std::vector<Layer> _pCapas;    /* Vector con cada una de las capas */
    double             _dEta;      /* Tasa de aprendizaje */
    double             _dMu;       /* Factor de momento */
    bool               _bSesgo;    /* ¿Van a tener sesgo las neuronas? */
  };
}

#endif
