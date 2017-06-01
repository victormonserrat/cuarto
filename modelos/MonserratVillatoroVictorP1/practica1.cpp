#include "IMC_options.hpp"
#include "IMC_DataBase.hpp"
#include "IMC_MultilayerPerceptron.hpp"
#include "IMC_statist.hpp"

#include <string>
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  // Inicializar valores por defecto
  std::string tvalue, Tvalue;
  int iteraciones = 1000, numCapas = 1, numNeuronas = 5;
  float eta = 0.1, mu = 0.9;
  bool bflag = false;
  // Procesar la línea de comandos
  if (imc::getOpt(argc, argv, tvalue, Tvalue, iteraciones, numCapas, numNeuronas, eta, mu, bflag) == EXIT_SUCCESS) {
    // Leer las bases de datos
    imc::DataBase pDatosTrain = imc::DataBase(tvalue);
    imc::DataBase pDatosTest = imc::DataBase(Tvalue);
    // Inicializar el vector "topología"
    // (número de neuronas por cada capa, incluyendo la de entrada
    //  y la de salida)
    std::vector<int> topologia;
    topologia.push_back(pDatosTrain.nNumEntradas());
    for (int i = 1; i < numCapas+1; i++) {
      topologia.push_back(numNeuronas);
    }
    topologia.push_back(pDatosTrain.nNumSalidas());
    // Inicializar perceptron
    imc::MultilayerPerceptron mlp;
    // Sesgo
    mlp.bSesgo(bflag);
    // Eta
    mlp.dEta(eta);
    // Mu
    mlp.dMu(mu);
    // Inicialización propiamente dicha
    mlp.inicializar(numCapas+2, topologia);
    // Semilla de los números aleatorios
    std::vector<int> semillas = {10, 20, 30, 40, 50};
    std::vector<double> erroresTest(5);
    std::vector<double> erroresTrain(5);
    for (size_t i = 0; i < 5; i++) {
      srand(semillas[i]);
      std::cout << "**********" << std::endl;
      std::cout << "SEMILLA " << semillas[i] << std::endl;
      std::cout << "**********" << std::endl;
      mlp.ejecutarAlgoritmoOnline(pDatosTrain, pDatosTest, iteraciones, erroresTrain[i], erroresTest[i]);
      std::cout << "Finalizamos => Error de test final: " << erroresTest[i] << std::endl;
    }
    std::cout << "HEMOS TERMINADO TODAS LAS SEMILLAS" << std::endl;
    // Calcular media y desviación típica de los errores de Train y de Test
    double mediaErrorTrain = imc::mean(erroresTrain);
    double desviacionTipicaErrorTrain = imc::sd(erroresTrain, mediaErrorTrain);
    double mediaErrorTest = imc::mean(erroresTest);
    double desviacionTipicaErrorTest = imc::sd(erroresTest, mediaErrorTest);
    std::cout << "INFORME FINAL" << std::endl;
    std::cout << "*************" << std::endl;
    std::cout << "Error de entrenamiento (Media +- DT): " << mediaErrorTrain << " +- " << desviacionTipicaErrorTrain << std::endl;
    std::cout << "Error de test (Media +- DT):          " << mediaErrorTest << " +- " << desviacionTipicaErrorTest << std::endl;
    return EXIT_SUCCESS;
  } else {
    imc::help();
    return EXIT_FAILURE;
  }
}
