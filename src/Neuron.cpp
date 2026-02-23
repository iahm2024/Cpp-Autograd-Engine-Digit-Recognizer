#include "../include/Neuron.h"

using namespace std;

Neuron::Neuron(int nin)
{
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int i = 0; i < nin; ++i)
    {
        weights.push_back(Value::newParameter(dis(gen)));
    }
    bias = Value::newParameter(0.0);
}

Neuron::~Neuron() {
    for (Value* weight : weights) {
        delete weight; 
    }

    weights.clear();

    if (bias) {
        delete bias;
    }
}

// Forward Pass
Value *Neuron::operator()(const vector<Value *> &x)
{
    // sum = w * x + b
    Value *activation = bias;
    for (size_t i = 0; i < weights.size(); ++i)
    {
        Value *product = (*weights[i]) * x[i];
        activation = (*activation) + product;
    }

    return activation->tanh();
}

vector<Value *> Neuron::parameters()
{
    std::vector<Value *> p = weights;
    p.push_back(bias);
    return p;
}