#ifndef VALUE_H
#define VALUE_H

#include <functional>
#include "GraphViz.h"
#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

class Value{
    public:
        Value() : data(0.0), grad(0.0), _backward([](){}) {}
        Value(double dataval) : data(dataval), grad(0.0), _backward([](){}) {}
        ~Value(){}
        
        function<void()> _backward;

        friend ostream& operator <<(ostream& out, const Value* val);
        Value* operator +(Value* val2);
        Value* operator +(double val2);
        Value* operator -(Value* val2);
        Value* operator -(double val2);
        Value* operator *(Value* val2);
        Value* operator *(double val2);

        void backward();
        Value* tanh();

        static Value* newValue(double data);
        static Value* newValue();

        static Value* newParameter(double data);

        static void clear_garbage_pool();

        double getdata() { return data; }
        void setdata(double v) {data = v;}

        double getgrad() { return grad; }
        void zero_grad() { grad = 0.0; }
        void setGrad(double val) { grad = val; }
        void addGrad(double val) { grad += val; }

        friend void draw_dot(Value* root, std::string filename);
        friend void trace(Value* v, std::set<Value*>& nodes, std::vector<std::pair<Value*, Value*>>& edges);
        

    private:
        bool is_permanent = false;
        double data, grad;
        vector<Value*> prev;
        static vector<Value*> garbage_pool;
        string operation;
        string label;
};

#endif