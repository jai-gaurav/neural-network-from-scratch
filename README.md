## Overview
This repository is to learn the math behind Neural Networks and make a neural network from scratch while following the Sentdex Neural Networks from Scratch Youtube series. The concepts covered in each practice .py file are listed below - 

## Contents
1. data_generator.py - Python script to create n spiral dataset used for training and testing
    - Uses the same code as the YT tutorial to generate the spiral dataset
    - For training we are using 100 feature sets for 3 classes
2. pratice1.py - Creating a simple neural network layer
    - Create a Layer class with weights and biases to allow the initalization of a layer
    - Create a forward method that takes inputs and computes the matrix product to find the output values of the layer
    - Allow for passing of a batch of inputs in the forward method to compute multiple output values simultaneously to create parallelization
3. practice2.py - Create ReLU activation function
    - Create a class for ReLU activation function and use it to compute outputs after getting outputs from a layer
    - Test the function using spiral data