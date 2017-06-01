CXXFLAGS += -Wall -Wextra -std=c++17 -O3
DEBUG += -g

all: practica1

clean:
	rm *.o practica1

IMC_DataBase.o: IMC_DataBase.cpp IMC_DataBase.hpp
	$(CXX) $(CXXFLAGS) -c IMC_DataBase.cpp

IMC_Neuron.o: IMC_Neuron.cpp IMC_Neuron.hpp
	$(CXX) $(CXXFLAGS) -c IMC_Neuron.cpp

IMC_Layer.o: IMC_Layer.cpp IMC_Layer.hpp
	$(CXX) $(CXXFLAGS) -c IMC_Layer.cpp

IMC_MultilayerPerceptron.o: IMC_MultilayerPerceptron.cpp IMC_MultilayerPerceptron.hpp
	$(CXX) $(CXXFLAGS) -c IMC_MultilayerPerceptron.cpp

practica1: IMC_MultilayerPerceptron.o IMC_Layer.o IMC_Neuron.o IMC_DataBase.o IMC_options.hpp practica1.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@
