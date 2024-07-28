# import libraries
import numpy as np
from datasets import spiral_data


# create a class to represent a layer
class Layer_Dense:
    # initialize the weights and biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randint(-999,1000, size=(n_inputs, n_neurons)) * 0.001
        self.biases = np.zeros((1, n_neurons))

    # forward pass by multiplying inputs by weights and adding biases
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# create a class to implement ReLU activation function
class Activation_ReLU:
    # forward pass by applying ReLU activation function
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# import spiral dataset
X, y = spiral_data(100, 3)


# create a dense layer with 2 input features and 3 output values
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

# forward pass
layer1.forward(X)
activation1.forward(layer1.output)

# print the output of the first few samples
print(layer1.output)
print(activation1.output)