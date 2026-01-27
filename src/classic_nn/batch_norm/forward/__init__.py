"""
Batch normalization forward propagation implementation.
"""
import numpy as np
from ...math import sigmoid, relu, tanh
from ...utils import get_num_layers


def linear_forward(A, W, G, B):
    """
    Implement the linear part of a layer's forward propagation.
    
    Arguments:
        A: activations from previous layer (or input data)
        W: weights matrix
        G: gamma matrix in batch normalization
        B: beta matrix in batch normalization
    
    Returns:
        Z: the input of the activation function
        cache: a python tuple containing "linear_cache" and "activation_cache"
    """
    Z = np.dot(W, A)
    mean_z = np.mean(Z, axis=1, keepdims=True)
    var_z = np.var(Z, axis=1, keepdims=True)
    epsilon = 1e-8
    Z_norm = (Z - mean_z) / np.sqrt(var_z + epsilon)
    Z = G * Z_norm + B
    cache = (A, W, G, B, Z_norm, mean_z, var_z, epsilon)
    return Z, cache


def linear_activation_forward(A_prev, W, G, B, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    
    Arguments:
        A_prev: activations from previous layer (or input data)
        W: weights matrix
        G: gamma matrix in batch normalization
        B: beta matrix in batch normalization
        activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
        A: the output of the activation function
        cache: a python tuple containing "linear_cache" and "activation_cache"
    """
    if activation == 'linear':
        Z, linear_cache = linear_forward(A_prev, W, G, B)
        A, activation_cache = Z, None
    elif activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, G, B)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, G, B)
        A, activation_cache = relu(Z)
    elif activation == 'tanh':
        Z, linear_cache = linear_forward(A_prev, W, G, B)
        A, activation_cache = tanh(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache


def custom_model_forward(X: np.ndarray, parameters: dict, layer_names: list[str], num_classes: int = 1, apply_sigmoid=False) -> tuple[np.ndarray, list]:
    """
    Implement custom model for classification
    
    Arguments:
        X: data, numpy array of shape (input size, number of examples)
        parameters: output of initialize_parameters_deep()
        layer_names: list of layer names
    
    Returns:
        AL: last post-activation value
        caches: list of caches containing:
                every cache of linear_activation_forward() (L-1 caches)
    """
    caches = []
    A = X
    L = get_num_layers(parameters)
    
    # check layer_names length
    if len(layer_names) != L:
        raise ValueError(f"Layer names length {len(layer_names)} does not match parameters length {L}")
    
    for l in range(1, L + 1):
        A_prev = A

        intented_activation = layer_names[l -1]
        
        if l == L and apply_sigmoid:
            current_activation = 'sigmoid'
        else:
            current_activation = intented_activation
        
        # validate the activation
        if current_activation not in ['relu', 'tanh', 'sigmoid', 'linear']:
            raise ValueError(f"Unsupported activation: {current_activation}")

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['G' + str(l)], parameters['B' + str(l)], current_activation)
        caches.append(cache)
        
    if num_classes is not None and A.shape[0] != num_classes:
        raise ValueError(f"Output layer size {A.shape[0]} does not match num_classes {num_classes}")
    
    return A, caches
