#ifndef NEURON_H
#define NEURON_H

#include "Value.h"
#include <vector>
#include <random>

using namespace std;

class Neuron
{
public:
    Neuron(int nin);
    ~Neuron();

    // Forward Pass
    Value *operator()(const vector<Value *> &x);

    vector<Value *> parameters();

private:
    vector<Value *> weights;
    Value *bias;
};

#endif