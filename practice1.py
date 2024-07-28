# import libraries
import numpy as np


# create a class to represent a layer
class Layer:
    # initialize the weights and biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randint(-999,1000, size=(n_inputs, n_neurons)) * 0.001
        self.biases = np.zeros((1, n_neurons))

    # forward pass by multiplying inputs by weights and adding biases
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# test the layer class
input_layer = Layer(4, 5)
hidden_layer_1 = Layer(5, 5)
hidden_layer_2 = Layer(5, 5)
output_layer = Layer(5, 2)

# single input
input_data = np.array([34, 13, 57, 62])

input_layer.forward(input_data)
hidden_layer_1.forward(input_layer.output)
hidden_layer_2.forward(hidden_layer_1.output)
output_layer.forward(hidden_layer_2.output)

print(output_layer.output)

# batch of inputs
input_data = np.array([[34, 13, 57, 62],
                        [23, 56, 78, 12],
                        [89, 32, 12, 34],
                        [12, 34, 56, 78]])

input_layer.forward(input_data)
hidden_layer_1.forward(input_layer.output)
hidden_layer_2.forward(hidden_layer_1.output)
output_layer.forward(hidden_layer_2.output)

print(output_layer.output)