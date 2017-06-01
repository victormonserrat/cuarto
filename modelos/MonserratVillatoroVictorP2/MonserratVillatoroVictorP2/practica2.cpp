#include "IMC_options.hpp"
#include "IMC_DataBase.hpp"
#include "IMC_MultilayerPerceptron.hpp"
#include "IMC_statist.hpp"

#include <string>
#include <iostream>
#include <vector>

#define N_SEMILLAS 5

int main(int argc, char *argv[]) {
  // Inicializar valores por defecto
  std::string tvalue, Tvalue;
  int iteraciones = 1000, numCapas = 1, numNeuronas = 5;
  int funcionError = 0;
  float eta = 0.1, mu = 0.9;
  bool bflag = false, oflag = false, sflag = false;
  // Procesar la línea de comandos
  if (imc::getOpt(argc, argv, tvalue, Tvalue, iteraciones, numCapas, numNeuronas, eta, mu, bflag, oflag, funcionError, sflag) == EXIT_SUCCESS) {
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
    // SigmoideCapaSalida
    mlp.bSigmoideCapaSalida(!sflag);
    // On-line
    mlp.bOnLine(oflag);
    // Eta
    if (!oflag) {
      eta /= pDatosTrain.nNumPatrones();
    }
    mlp.dEta(eta);
    // Mu
    mlp.dMu(mu);
    // Inicialización propiamente dicha
    mlp.inicializar(numCapas+2, topologia);
    // Semilla de los números aleatorios
    std::vector<int> semillas = {10, 20, 30, 40, 50};
    std::vector<double> erroresTest(N_SEMILLAS);
    std::vector<double> erroresTrain(N_SEMILLAS);
    std::vector<double> ccrs(N_SEMILLAS);
    std::vector<double> ccrsTrain(N_SEMILLAS);
    for (size_t i = 0; i < N_SEMILLAS; i++) {
      srand(semillas[i]);
      std::cout << "**********" << std::endl;
      std::cout << "SEMILLA " << semillas[i] << std::endl;
      std::cout << "**********" << std::endl;
      mlp.ejecutarAlgoritmo(pDatosTrain, pDatosTest, iteraciones, erroresTrain[i], erroresTest[i], ccrsTrain[i], ccrs[i], funcionError);
      std::cout << "Finalizamos => CCR de test final: " << ccrs[i] << std::endl;
    }
    std::cout << "HEMOS TERMINADO TODAS LAS SEMILLAS" << std::endl;
    // Calcular media y desviación típica de los errores de Train y de Test
    double mediaErrorTrain = imc::mean(erroresTrain);
    double desviacionTipicaErrorTrain = imc::sd(erroresTrain, mediaErrorTrain);
    double mediaErrorTest = imc::mean(erroresTest);
    double desviacionTipicaErrorTest = imc::sd(erroresTest, mediaErrorTest);
    double mediaCCRTrain = imc::mean(ccrsTrain);
    double desviacionTipicaCCRTrain = imc::sd(ccrsTrain, mediaCCRTrain);
    double mediaCCR = imc::mean(ccrs);
    double desviacionTipicaCCR = imc::sd(ccrs, mediaCCR);
    std::cout << "INFORME FINAL" << std::endl;
    std::cout << "*************" << std::endl;
    std::cout << "Error de entrenamiento (Media +- DT): " << mediaErrorTrain << " +- " << desviacionTipicaErrorTrain << std::endl;
    std::cout << "Error de test (Media +- DT): " << mediaErrorTest << " +- " << desviacionTipicaErrorTest << std::endl;
    std::cout << "CCR de entrenamiento (Media +- DT): " << mediaCCRTrain << " +- " << desviacionTipicaCCRTrain << std::endl;
    std::cout << "CCR de test (Media +- DT): " << mediaCCR << " +- " << desviacionTipicaCCR << std::endl;
    return EXIT_SUCCESS;
  } else {
    imc::help();
    return EXIT_FAILURE;
  }
}
