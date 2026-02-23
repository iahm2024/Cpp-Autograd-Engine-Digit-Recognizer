CC = g++
CFLAGS = -std=c++17 -g
CORE = src/Value.cpp src/Neuron.cpp src/Layer.cpp src/MLP.cpp src/GraphViz.cpp src/DataLoader.cpp

all: digits

digits: Digits.cpp $(CORE)
	$(CC) $(CFLAGS) $^ -o digits

clean:
	rm -f *_test