"""
Math functions for neural networks.
"""
import numpy as np

def sigmoid(Z):
    """
    Implement the sigmoid activation function.
    
    Arguments:
        Z: numpy array of any shape
    
    Returns:
        A: the output of the sigmoid function
        cache: a python tuple containing "Z" for backward propagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu(Z):
    """
    Implement the ReLU activation function.
    
    Arguments:
        Z: numpy array of any shape
    
    Returns:
        A: the output of the ReLU function
        cache: a python tuple containing "Z" for backward propagation
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def tanh(Z):
    """
    Implement the tanh activation function.
    
    Arguments:
        Z: numpy array of any shape
    
    Returns:
        A: the output of the tanh function
        cache: a python tuple containing "Z" for backward propagation
    """
    A = np.tanh(Z)
    cache = Z
    return A, cache
