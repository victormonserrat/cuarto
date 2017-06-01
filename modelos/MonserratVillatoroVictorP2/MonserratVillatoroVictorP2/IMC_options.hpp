#ifndef __IMC_OPTIONS_HPP__
#define __IMC_OPTIONS_HPP__

#include <cstdlib>
#include <unistd.h>
#include <string>
#include <iostream>
#include <fstream>

namespace imc {
  /* Imprimir ayuda */
  void help() {
    std::cout << std::endl << "\e[34m-t: Indica el nombre del fichero que contiene los datos de entrenamiento a utilizar. Sin este argumento, el programa no puede funcionar" << std::endl << std::endl;
    std::cout << "-T: Indica el nombre del fichero que contiene los datos de test a utilizar. Si no se especifica este argumento, se utilizan los datos de entrenamiento como test." << std::endl << std::endl;
    std::cout << "-i: Indica el número de iteraciones del bucle externo a realizar. Por defecto, 1000 iteraciones." << std::endl << std::endl;
    std::cout << "-l: Indica el número de capas ocultas del modelo de red neuronal. Por defecto, 1 capa oculta." << std::endl << std::endl;
    std::cout << "-h: Indica el número de neuronas a introducir en cada una de las capas ocultas. Por defecto, 5 neuronas." << std::endl << std::endl;
    std::cout << "-e: Indica el valor del parámetro eta (η). Por defecto, η = 0,1." << std::endl << std::endl;
    std::cout << "-m: Indica el valor del parámetro mu (μ). Por defecto, μ = 0,9." << std::endl << std::endl;
    std::cout << "-b: Indica si se va a utilizar sesgo en las neuronas. Por defecto. Si no se especifica este argumento, no se utiliza sesgo." << std::endl << std::endl;
    std::cout << "-o: Indica si se va a utilizar la versión on-line. Si no se especifica, utilizaremos la versión off-line." << std::endl << std::endl;
    std::cout << "-f: Indica la función de error que se va a utilizar durante el aprendizaje (0 para el error MSE y 1 para la entropı́a cruzada). Por defecto, utilizar el error MSE." << std::endl << std::endl;
    std::cout << "-s: Indica si vamos utilizar la función softmax en la capa de salida. Si no se especifica, utilizaremos la función sigmoide.\e[0m" << std::endl << std::endl;
  }
  /* Recoger argumentos por línea de comandos */
  int getOpt(int argc, char *argv[], std::string &t, std::string &T, int &i, int &l, int &h, float &e, float &m, bool &b, bool &o, int &f, bool &s) {
    bool tFlag = false, TFlag = false;
    int c;
    while ((c = getopt(argc, argv, "at:T:i:l:h:e:m:bof:s")) != -1) {
      switch (c) {
        // Ejecutar ayuda
        case 'a':
          return EXIT_FAILURE;
          break;
        // Leer fichero de train y de test desde la línea de comandos
        case 't':
          tFlag = true;
          t = optarg;
          break;
        case 'T':
          TFlag = true;
          T = optarg;
          break;
        // Leer iteraciones, capas y neuronas desde la línea de comandos
        case 'i':
          i = atoi(optarg);
          break;
        case 'l':
          l = atoi(optarg);
          break;
        case 'h':
          h = atoi(optarg);
          break;
        // Leer sesgo, eta y mu de la línea desde comandos
        case 'e':
          e = atof(optarg);
          break;
        case 'm':
          m = atof(optarg);
          break;
        case 'b':
          b = true;
          break;
        case 'o':
          o = true;
          break;
        case 'f':
          f = atoi(optarg);
          break;
        case 's':
          s = true;
          break;
        case '?':
          return EXIT_FAILURE;
        default:
          abort();
      }
    }
    bool failure = false;
    if (!tFlag) {
      std::cout << std::endl << "\e[31mError: debe especificar el fichero de entrenamiento.\e[0m" << std::endl;
      failure = true;
    } else {
      std::ifstream f(t.c_str());
      if (!f.good()) {
        std::cout << std::endl << "\e[31mError: el fichero de entrenamiento no existe.\e[0m" << std::endl;
        failure = true;
      }
    }
    if (!TFlag) {
        T = t;
    } else {
      std::ifstream f(T.c_str());
      if (!f.good()) {
        std::cout << std::endl << "\e[31mError: el fichero de test no existe.\e[0m" << std::endl;
        failure = true;
      }
    }
    if (i < 1) {
      std::cout << std::endl << "\e[31mError: el número de iteraciones debe ser positivo.\e[0m" << std::endl;
      failure = true;
    }
    if (l < 1) {
      std::cout << std::endl << "\e[31mError: el número de capas ocultas debe ser positivo.\e[0m" << std::endl;
      failure = true;
    }
    if (h < 1) {
      std::cout << std::endl << "\e[31mError: el número de neuronas por capa oculta debe ser positivo.\e[0m" << std::endl;
      failure = true;
    }
    if (e < 0) {
      std::cout << std::endl << "\e[31mError: eta debe tener un valor positivo.\e[0m" << std::endl;
      failure = true;
    }
    if (m < 0) {
      std::cout << std::endl << "\e[31mError: mu debe tener un valor positivo.\e[0m" << std::endl;
      failure = true;
    }
    if (f < 0 || f > 1) {
      std::cout << std::endl << "\e[31mError: función de error " << f << " no válida.\e[0m" << std::endl;
      failure = true;
    }
    if (failure) {
      return EXIT_FAILURE;
    } else {
      return EXIT_SUCCESS;
    }
  }
}

#endif
