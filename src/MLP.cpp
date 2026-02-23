#include "../include/MLP.h"

using namespace std;

MLP::MLP(int nin, vector<int> nouts)
{
    vector<int> sz = {nin};
    sz.insert(sz.end(), nouts.begin(), nouts.end());

    for (size_t i = 0; i < nouts.size(); ++i)
    {
        layers.push_back(new Layer(sz[i], sz[i + 1]));
    }
}

MLP::~MLP()
{
    for (Layer *layer : layers)
    {
        delete layer;
    }

    layers.clear();
}

vector<Value *> MLP::operator()(vector<Value *> x)
{
    for (auto &layer : layers)
    {
        x = (*layer)(x);
    }
    return x;
}

vector<Value *> MLP::parameters()
{
    vector<Value *> p;
    for (auto &layer : layers)
    {
        vector<Value *> lp = layer->parameters();
        p.insert(p.end(), lp.begin(), lp.end());
    }
    return p;
}

void MLP::save(const std::string &filename)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        // dump all weights and biases line by line into the file
        for (auto p : parameters())
        {
            file << p->getdata() << "\n";
        }
        file.close();
        std::cout << "Model saved to: " << filename << std::endl;
    }
    else
    {
        std::cout << "Error: Could not open file for saving!" << std::endl;
    }
}

void MLP::load(const std::string &filename)
{
    std::ifstream file(filename);
    if (file.is_open())
    {
        std::vector<Value *> params = parameters();
        double val;
        int count = 0;

        // read values one by one and overwrite current model weights
        while (file >> val && count < params.size())
        {
            params[count]->setdata(val);
            count++;
        }

        // safety check just in case we loaded a file meant for a different network size
        if (count != params.size())
        {
            std::cout << "File data doesn't match current model architecture!" << std::endl;
        }
        else
        {
            std::cout << "Model loaded from: " << filename << std::endl;
        }
        file.close();
    }
    else
    {
        std::cout << " Could not find or open file: " << filename << std::endl;
    }
}