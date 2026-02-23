#include "../include/Layer.h"

using namespace std;

Layer::Layer(int nin, int nout)
{
    for (int i = 0; i < nout; ++i)
    {
        neurons.push_back(new Neuron(nin));
    }
}

Layer::~Layer() {
    for (Neuron* neuron : neurons) {
        delete neuron;
    }
    neurons.clear();
}

vector<Value *> Layer::operator()(const vector<Value *> &x)
{
    vector<Value *> outputs;
    for (auto &n : neurons)
    {
        outputs.push_back((*n)(x));
    }
    return outputs;
}

vector<Value *> Layer::parameters()
{
    vector<Value *> p;
    for (auto &n : neurons)
    {
        vector<Value *> np = n->parameters();
        p.insert(p.end(), np.begin(), np.end());
    }
    return p;
}
