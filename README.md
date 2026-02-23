# 🧠 C++ Autograd Engine & Digit Recognizer

A lightweight, scalar-based autograd engine and Multi-Layer Perceptron (MLP) built completely from scratch in C++. I created this project to deeply understand how backpropagation, computation graphs, and neural networks work under the hood without relying on black-box libraries like PyTorch or TensorFlow.

To prove that the math works, I trained this engine on the UCI Digits dataset (a smaller version of MNIST), and it successfully recognizes handwritten digits with **~89.5% accuracy**.

## ✨ Features
* **Zero Dependencies:** No external math or matrix libraries (like Eigen) were used. 
* **Custom Autograd:** Builds a dynamic computation graph using pointers and calculates gradients automatically.
* **No Memory Leaks:** Strictly managed memory. Verified with Valgrind (`0 errors from 0 contexts`, `0 bytes leaked`).
* **Interactive Demo:** Converts pixel data into ASCII art in the terminal to show what the network is "seeing" alongside its prediction.

## 🚀 Getting Started

### 1. Build the project
I included a simple Makefile. Just run:
```bash
make
./digits
```
Note: You can easily switch between training a new model or loading the pre-trained weights (model_digits.txt) by changing the do_training boolean inside main().

🛠️ Engineering Note: Why is it slow?

If you train this model from scratch, you will notice that the epochs take some time. This is expected and is actually a great lesson in computer architecture!

This engine is Scalar-Based. Every single weight and bias is its own object (new Value(0.5)) stored somewhere in the RAM, connected by pointers. During the forward and backward passes, the CPU has to chase these pointers around memory (Pointer Chasing). Modern frameworks like PyTorch use contiguous memory blocks (Matrices/Tensors) and compute them in parallel using SIMD or GPUs.

So, while this is mathematically identical to a real neural network, it is engineered for education and clarity, not performance.
📝 Future Improvements

    Implement mini-batch gradient descent.

    Add different activation functions (like ReLU) and a Softmax output layer.

    Move from a scalar approach to a matrix-based approach to speed up training.

Built for learning purposes. Inspired by Andrej Karpathy's micrograd.
