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
    bool bSigmoideCapaSalida() const {return _bSigmoideCapaSalida;}
    void bSigmoideCapaSalida(bool const &b) {_bSigmoideCapaSalida = b;}
    bool bOnLine() const {return _bOnLine;}
    void bOnLine(bool const &b) {_bOnLine = b;}

    // Obtener un número entero aleatorio en el intervalo [Low,High]
    int enteroAleatorio(int const &Low, int const &High);
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
    // funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
    double calcularErrorSalida(std::vector<double> const &target, int const &funcionError);
    // Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
    // funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
    void retropropagarError(std::vector<double> const &objetivo, int const &funcionError);
    // Acumular los cambios producidos por un patrón en deltaW
    void acumularCambio();
    // Actualizar los pesos de la red, desde la primera capa hasta la última
    void ajustarPesos();
    // Imprimir la red, es decir, todas las matrices de pesos
    void imprimirRed();
    // Simular la red: propagar las entradas hacia delante, retropropagar el error y ajustar los pesos
    // entrada es el vector de entradas del patrón y objetivo es el vector de salidas deseadas del patrón
    // El paso de ajustar pesos solo deberá hacerse si el algoritmo es on-line
    // Si no lo es, el ajuste de pesos hay que hacerlo en la función "entrenar"
    // funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
    void simularRed(std::vector<double> const &entrada, std::vector<double> const &objetivo, int const &funcionError);
    // Entrenar la red on-line para un determinado fichero de datos (pasar una vez por todos los patrones)
    // Si es offline, después de pasar por ellos hay que ajustar pesos. Sino, ya se ha ajustado en cada patrón
    void entrenar(DataBase const &pDatosTrain, int const &funcionError);
    // Probar la red con un conjunto de datos y devolver el error MSE cometido
    // funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
    double test(DataBase const &pDatosTest, int const &funcionError);
    // Probar la red con un conjunto de datos y devolver el error CCR cometido
    double testClassification(DataBase const &pDatosTest);
    // Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
    // Una vez terminado, probar como funciona la red en pDatosTest
    // Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
    // funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
    void ejecutarAlgoritmo(DataBase const &pDatosTrain, DataBase const &pDatosTest, int const &maxiter, double &errorTrain, double &errorTest, double &ccrTrain, double &ccrTest, int const &funcionError);

  private:
    int                _nNumCapas; /* Número de capas total en la red */
    std::vector<Layer> _pCapas;    /* Vector con cada una de las capas */
    double             _dEta;      /* Tasa de aprendizaje */
    double             _dMu;       /* Factor de momento */
    bool               _bSesgo;    /* ¿Van a tener sesgo las neuronas? */
    bool               _bSigmoideCapaSalida;
    bool               _bOnLine;
  };
}

#endif
