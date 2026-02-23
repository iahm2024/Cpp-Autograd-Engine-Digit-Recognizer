#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "Value.h"

struct Dataset {
    std::vector<std::vector<Value*>> inputs;
    std::vector<std::vector<Value*>> targets;
};

// One-Hot Encoding Yapar
std::vector<Value*> get_target_vector(int label);

Dataset load_csv(const std::string& filename);

#endif