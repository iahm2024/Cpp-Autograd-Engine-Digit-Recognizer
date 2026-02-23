#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

using namespace std;

class Layer {
public:
    Layer(int nin, int nout);
    ~Layer();

    vector<Value*> operator()(const std::vector<Value*>& x);

    vector<Value*> parameters();

private:
    vector<Neuron*> neurons;
};

#endif