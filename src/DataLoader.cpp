#include "../include/DataLoader.h"

std::vector<Value*> get_target_vector(int label) {
    std::vector<Value*> t;
    for(int i=0; i<10; i++) {
        if (i == label) 
            t.push_back(new Value(1.0)); 
        else 
            t.push_back(new Value(-1.0));
    }
    return t;
}

Dataset load_csv(const std::string& filename) {
    Dataset data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << " Could not find or open file: " << filename << std::endl;
        exit(1);
    }

    int count = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val_str;
        std::vector<Value*> row_input;

        for (int i = 0; i < 64; ++i) {
            std::getline(ss, val_str, ',');
            double pixel = std::stod(val_str);
            
            row_input.push_back(new Value(pixel / 16.0)); 
        }

        std::getline(ss, val_str, ',');
        int label = std::stoi(val_str);

        data.inputs.push_back(row_input);
        data.targets.push_back(get_target_vector(label));
        
        count++;
    }

    std::cout << count << " data points loaded successfully" << std::endl;
    return data;
}