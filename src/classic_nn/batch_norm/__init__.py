"""
Classic batch normalization implementation.
"""

from .. import optimizers
from ..utils import get_num_layers
from ..loss import BCE_WithLogitsLoss
from .forward import custom_model_forward
from .backward import custom_model_backward


import numpy as np


def initialize_parameters_deep(layer_dims, optimizer_instance: optimizers.OptimizerFactory=None):
    """
    Initialize parameters for deep neural network, including batch normalization parameters. 
    
    Args:
        layer_dims: list of integers representing the number of units in each layer
    
    Returns:
        Dictionary containing initialized parameters
        W1, G1, B1, W2, G2, B2, ...
    """    
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 1/np.sqrt(layer_dims[l-1])        
        parameters['G' + str(l)] = np.ones((layer_dims[l], 1))
        parameters['B' + str(l)] = np.zeros((layer_dims[l], 1))
    
    assert(parameters['W' + str(1)].shape == (layer_dims[1], layer_dims[0]))    
    assert(parameters['G' + str(1)].shape == (layer_dims[1], 1))
    assert(parameters['B' + str(1)].shape == (layer_dims[1], 1))
    
    if optimizer_instance is not None:
        print(f"The model is using optimizer {optimizer_instance}")
        parameters['_batch_norm'] = True
        optimizer_instance.initialize_parameters(parameters, layer_dims)
    
    return parameters   



def gradient_descent(X, Y, parameters, activations, learning_rate=0.0075, num_classes=1, last_activation="sigmoid", lambda_reg=0.01):
    """
    Running forward and backward propagation on a single batch of data
    Returns cost, grads, parameters
    """    
    A, caches = custom_model_forward(X, parameters, activations, num_classes, apply_sigmoid=(last_activation == "sigmoid"))
    cost = BCE_WithLogitsLoss(A, Y, parameters, from_logits=(last_activation == "linear"), lambda_reg=lambda_reg)
    grads = custom_model_backward(A, Y, caches, activations, last_activation, lambda_reg)
    parameters = update_parameters(parameters, grads, learning_rate)
    
    return cost, grads, parameters


def update_parameters(parameters, grads, learning_rate ):
    """
    Update parameters using gradient descent
    
    Arguments:
        parameters: dictionary containing your parameters 
        grads: dictionary containing your gradients, output of L_model_backward
    
    Returns:
        parameters: dictionary containing your updated parameters 
    """
    L = get_num_layers(parameters)
    optimizer_instance = parameters.get('_optimizer_instance', None)

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        if optimizer_instance is not None:
            optimizer_instance.update_parameters_once(
                parameters, 
                grads,
                l + 1,
                learning_rate
            )            
        else:
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["G" + str(l+1)] = parameters["G" + str(l+1)] - learning_rate * grads["dG" + str(l+1)]
            parameters["B" + str(l+1)] = parameters["B" + str(l+1)] - learning_rate * grads["dB" + str(l+1)]
    
    return parameters


def gradient_check(parameters, gradients, X, Y, epsilon=1e-7):
    # TODO Implementation of gradient checking
    pass