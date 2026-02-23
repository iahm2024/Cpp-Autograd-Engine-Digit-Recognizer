#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include "../include/Value.h"
#include "../include/MLP.h"
#include "../include/DataLoader.h"

using namespace std;

// helper function to print the digit in the terminal and see what the model thinks
void show_prediction(MLP *model, const std::vector<Value *> &input, const std::vector<Value *> &target)
{
    cout << "\n-----------------------------------" << endl;
    cout << "        NETWORK VISION        " << endl;
    cout << "-----------------------------------" << endl;

    // draw the 8x8 image using ascii blocks
    for (int row = 0; row < 8; row++)
    {
        for (int col = 0; col < 8; col++)
        {
            // if pixel is somewhat dark, draw a block
            if (input[row * 8 + col]->getdata() > 0.3)
                cout << "██";
            else
                cout << "..";
        }
        cout << endl;
    }

    // get model prediction
    vector<Value *> out = (*model)(input);
    int predicted = 0;
    double max_val = -9999;

    for (int j = 0; j < 10; j++)
    {
        if (out[j]->getdata() > max_val)
        {
            max_val = out[j]->getdata();
            predicted = j;
        }
    }

    // get the actual label from one-hot target array
    int actual = 0;
    for (int j = 0; j < 10; j++)
    {
        if (target[j]->getdata() > 0)
            actual = j;
    }

    cout << "\n=> PREDICTION : [ " << predicted << " ]";
    if (predicted == actual)
        cout << "  [SUCCESS] -> Model got it right!" << endl;
    else
        cout << "  [FAIL] -> Model got it wrong! (Actual: " << actual << ")" << endl;
}

int main()
{
    // load the dataset
    cout << "Loading data..." << endl;
    Dataset data = load_csv("digits.csv");

    int train_size = 400;

    // init model
    // inputs: 64 (8x8 pixels)
    // hidden layer: 30 neurons
    // output: 10 neurons (digits 0-9)
    MLP *model = new MLP(64, {30, 10});
    
    // switch this to false if you just want to test the saved model
    bool do_training = false;

    // setup for shuffling data every epoch
    std::vector<size_t> indices(data.inputs.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());

    int epochs = 40;
    double learning_rate = 0.02;

    if (do_training)
    {
        cout << "Starting training..." << endl;

        for (int k = 0; k < epochs; k++)
        {
            // shuffle data indices so the model doesn't memorize the order
            std::shuffle(indices.begin(), indices.end(), g);

            double total_loss = 0;
            int correct_count = 0;

            for (size_t i = 0; i < indices.size(); i++)
            {
                int idx = indices[i];

                // forward pass
                vector<Value *> out = (*model)(data.inputs[idx]);

                Value *loss = new Value(0.0);

                int predicted = 0;
                double max_val = -9999;

                // calculate mse loss and find the predicted digit
                for (int j = 0; j < 10; j++)
                {
                    if (out[j]->getdata() > max_val)
                    {
                        max_val = out[j]->getdata();
                        predicted = j;
                    }

                    Value *diff = (*out[j]) - data.targets[idx][j];
                    loss = (*loss) + ((*diff) * diff);
                }

                // find the actual target digit
                int actual = 0;
                for (int j = 0; j < 10; j++)
                {
                    if (data.targets[idx][j]->getdata() > 0)
                        actual = j;
                }

                if (predicted == actual)
                    correct_count++;

                total_loss += loss->getdata();

                // zero out gradients before backward pass
                for (Value *p : model->parameters())
                {
                    p->zero_grad();
                }

                // backward pass (compute gradients)
                loss->backward();

                // update weights (gradient descent)
                for (auto p : model->parameters())
                {
                    p->setdata(p->getdata() - learning_rate * p->getgrad());
                }

                // memory cleanup for computation graph
                Value::clear_garbage_pool();
            }

            cout << "Epoch " << k
                 << " | Loss: " << total_loss / data.inputs.size()
                 << " | Accuracy: " << (double)correct_count / data.inputs.size() * 100 << "%"
                 << endl;

            // decay learning rate slightly to converge better
            learning_rate *= 0.95;
        }
        
        // save weights to a file after training
        model->save("model_digits.txt");
    }
    else
    {
        cout << "Loading saved model..." << endl;
        model->load("model_digits.txt");
    }

    cout << "\n--- TEST PHASE ---" << endl;
    int test_correct = 0;
    
    // test the model on unseen data
    for (size_t i = train_size; i < data.inputs.size(); i++)
    {
        vector<Value *> out = (*model)(data.inputs[i]);

        int predicted = 0;
        double max_val = -9999;
        
        for (int j = 0; j < 10; j++)
        {
            if (out[j]->getdata() > max_val)
            {
                max_val = out[j]->getdata();
                predicted = j;
            }
        }

        int actual = 0;
        for (int j = 0; j < 10; j++)
        {
            if (data.targets[i][j]->getdata() > 0)
                actual = j;
        }

        if (predicted == actual)
            test_correct++;
    }

    cout << "Test Accuracy: " << (double)test_correct / (data.inputs.size() - train_size) * 100 << "%" << endl;

    // show off a few random predictions visually
    cout << "\n--- INTERACTIVE DEMO ---" << endl;
    // pick some random indices from the end of the dataset
    show_prediction(model, data.inputs[1700], data.targets[1700]);
    show_prediction(model, data.inputs[1750], data.targets[1750]);
    show_prediction(model, data.inputs[1780], data.targets[1780]);

    // final memory cleanup
    Value::clear_garbage_pool();

    for (auto &row : data.inputs)
    {
        for (Value *val : row)
        {
            delete val;
        }
        row.clear();
    }
    data.inputs.clear();

    delete model;
    return 0;
}