#include "../include/Value.h"

using namespace std;

vector<Value*> Value::garbage_pool;

ostream& operator <<(ostream& out, const Value* val){
    
    out << "data : " << val->data << endl;
    out << "grad : " << val->grad << endl;

    return out;
}

Value* Value::operator +(Value* val2){

    Value* out = Value::newValue();
    out->data = this->data + val2->data;

    out->prev.push_back(this);
    out->prev.push_back(val2);

    out->_backward = [this, val2, out]() {
        this->grad += 1.0 * out->grad;
        val2->grad += 1.0 * out->grad;
    };

    out->operation = "+";

    return out;
}

Value* Value::operator +(double val2){
    
    Value* out = Value::newValue();
    out->data = this->data + val2;

    out->prev.push_back(this);

    out->_backward = [this, out]() {
        this->grad += 1.0 * out->grad;
    };

    out->operation = "+";

    return out;
}

Value* Value::operator -(Value* val2){
    Value* out = Value::newValue();
    out->data = this->data - val2->data;
    out->prev.push_back(this);
    out->prev.push_back(val2);

    out->_backward = [this, val2, out]() {
        this->grad += 1.0 * out->grad;
        val2->grad -= 1.0 * out->grad;
    };

    out->operation = "-";

    return out;
}

Value* Value::operator -(double val2){
    Value* out = Value::newValue();
    out->data = this->data - val2;
    
    out->prev.push_back(this);

    out->_backward = [this, out]() {
        this->grad += 1.0 * out->grad;
    };

    out->operation = "-";

    return out;
}

Value* Value::operator *(Value* val2){

    Value* out = Value::newValue();
    out->data = this->data * val2->data;

    out->prev.push_back(this);
    out->prev.push_back(val2);

    out->_backward = [this, val2, out]() {
        this->grad += val2->data * out->grad;
        val2->grad += this->data * out->grad;
    };

    out->operation = "*";

    return out;
}

Value* Value::operator *(double val2){
    
    Value* out = Value::newValue();
    out->data = this->data * val2;

    out->prev.push_back(this);

    out->_backward = [this, val2, out]() {
        this->grad += val2 * out->grad;
    };

    out->operation = "*";

    return out;
}

Value* Value::newValue(double data){
    
    Value* val = new Value(data);
    garbage_pool.push_back(val);
    return val;

}

Value* Value::newValue(){
    
    Value* val = new Value();
    garbage_pool.push_back(val);
    return val;

}

Value* Value::newParameter(double data) {
    Value* v = newValue(data);
    v->is_permanent = true;
    return v;
}

void Value::clear_garbage_pool() {
    std::vector<Value*> next_pool;
    
    for (Value* v : garbage_pool) {
        if (v->is_permanent) {
            next_pool.push_back(v);
        } else {
            delete v;
        }
    }
    garbage_pool = next_pool;
}

void Value::backward(){
    vector<Value*> topo;
    set<Value*> visited;

    function<void(Value*)> build_topo = [&](Value* v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (Value* child : v->prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(this);

    this->grad = 1.0;

    reverse(topo.begin(), topo.end());

    for (Value* v : topo) {
        v->_backward();
    } 
}

Value* Value::tanh(){

    Value* out = Value::newValue();
    out->prev.push_back(this);
    out->data = ((exp(2.0 * this->data) - 1) / (exp(2.0 * this->data) + 1));

    out->_backward = [this, out]() {
        this->grad += (1.0 - out->data * out->data) * out->grad;
    };

    out->operation = "tanh";

    return out;
}