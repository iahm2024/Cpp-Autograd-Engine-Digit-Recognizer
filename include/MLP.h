#ifndef MLP_H
#define MLP_H

#include "Layer.h"

using namespace std;

class MLP {
public:
    MLP(int nin, vector<int> nouts);
    ~MLP();

    vector<Value*> operator()(vector<Value*> x);

    vector<Value*> parameters();
    
    void save(const std::string &filename);
    void load(const std::string &filename);

private:
    vector<Layer*> layers;
};

#endif