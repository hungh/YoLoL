"""
Classic batch normalization implementation.
"""

# IN PROGRESS..... Do not use 

from classic_nn.regularization import L2_regularization 
from classic_nn.activation import sigmoid, relu, tanh

import numpy as np


def initialize_parameters_deep(layer_dims):
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
    
    return parameters   


    
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
    if activation == 'sigmoid':
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


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->(SIGMOID|LOGITS) computation
    
    Arguments:
        X: data, numpy array of shape (input size, number of examples)
        parameters: output of initialize_parameters_deep()
    
    Returns:
        AL: last post-activation value
        caches: list of caches containing:
                every cache of linear_activation_forward() (L-1 caches)
    """
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['G' + str(l)], parameters['B' + str(l)], 'relu')
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['G' + str(L)], parameters['B' + str(L)], 'sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches


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
    L = len(parameters) // 2

    # check layer_names length
    if len(layer_names) - 1 != L:
        raise ValueError(f"Layer names length {len(layer_names)} does not match parameters length {L}")
    
    for l in range(1, L + 1):
        A_prev = A

        activation = layer_names[l -1]
        if activation not in ['relu', 'tanh', 'sigmoid']:
            raise ValueError(f"Unsupported activation: {activation}")

        current_activation = 'sigmoid' if (l == L and apply_sigmoid) else 'relu'

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['G' + str(l)], parameters['B' + str(l)], current_activation)
        caches.append(cache)
        
    if num_classes is not None and A.shape[0] != num_classes:
        raise ValueError(f"Output layer size {A.shape[0]} does not match num_classes {num_classes}")
    
    return A, caches


# implementation of binary cross entropy logits loss (similar to torch.nn.BCEWithLogitsLoss)
def  BCE_WithLogitsLoss(AL, Y, parameters: dict, from_logits: bool = True, lambda_reg: float = 0.01) -> float:
    """
    Implement binary cross entropy logits loss
    
    Arguments:
        AL: probability vector, output of forward propagation
        Y: true "label" vector
    
    Returns:
        cost: cost value
    """
    # ensure AL and Y are the same shape if
    if AL.shape != Y.shape:
        if AL.shape == Y.T.shape:
            Y = Y.T
        else:
            raise ValueError(f"AL shape {AL.shape} does not match Y shape {Y.shape}")
    
    m = Y.shape[1]
    
    if from_logits:        
        AL = 1. / (1. + np.exp(-AL))
        AL = np.clip(AL, 1e-15, 1 - 1e-15)  
    
    # Compute the cross-entropy cost
    cost = -np.sum(Y * np.log(AL + 1e-15) + (1 - Y) * np.log(1 - AL + 1e-15)) / m
    
    # Make sure to reshape the cost to avoid nested arrays
    cost = np.squeeze(cost) + L2_regularization(m, parameters, lambda_reg)
    return cost


def linear_backward(dZ, cache, lambda_reg=None):
    """
    Arguments:
        dZ: Gradient of the cost with respect to the linear output
        cache: tuple of (A_prev, W, G, B, Z_norm, mean, var, epsilon) from forward pass
        lambda_reg: L2 regularization parameter
    
    Returns:
        dA_prev: Gradient w.r.t. previous layer's activations
        dW: Gradient w.r.t. weights W
        dG: Gradient w.r.t. scale parameter gamma
        dB: Gradient w.r.t. shift parameter beta
    """
    A_prev, W, G, B, Z_norm, mean, var, epsilon = cache
    m = A_prev.shape[1]
    
    # Gradients of batch norm parameters
    dG = np.sum(dZ * Z_norm, axis=1, keepdims=True) / m
    dB = np.sum(dZ, axis=1, keepdims=True) / m
    
    # Gradient through batch norm
    dZ_norm = dZ * G
    
    # Gradient of variance
    dvar = np.sum(dZ_norm * (A_prev - mean) * -0.5 * (var + epsilon)**(-1.5), axis=1, keepdims=True)
    
    # Gradient of mean
    dmean = np.sum(dZ_norm * -1 / np.sqrt(var + epsilon), axis=1, keepdims=True) + \
            dvar * np.sum(-2 * (A_prev - mean), axis=1, keepdims=True) / m
    
    # Gradient of Z (before batch norm)
    dZ_prev = (dZ_norm / np.sqrt(var + epsilon)) + \
              (dvar * 2 * (A_prev - mean) / m) + \
              (dmean / m)
    
    # Gradient of weights
    dW = np.dot(dZ_prev, A_prev.T) / m
    if lambda_reg is not None:
        dW += (lambda_reg / m) * W
    
    # Gradient w.r.t. previous layer's activations
    dA_prev = np.dot(W.T, dZ_prev)
    
    return dA_prev, dW, dG, dB


def relu_backward(dA, cache):
    """
    Implement the backward propagation for the RELU function with batch norm.
    
    Arguments:
        dA: post-activation gradient, of shape (n, m)
        cache: tuple containing (Z, linear_cache) where:
            - Z is the input to the activation function
            - linear_cache is the cache from the linear_forward pass
              (A_prev, W, G, B, Z_norm, mean, var, epsilon)
    
    Returns:
        dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1)
        dW: Gradient of the cost with respect to W (current layer l)
        dG: Gradient of the cost with respect to the scale parameter gamma
        dB: Gradient of the cost with respect to the shift parameter beta
    """
    Z, linear_cache = cache
    dZ = np.array(dA, copy=True)  # Just converting dA to a correct object
    
    # When z <= 0, set dz to 0
    dZ[Z <= 0] = 0
    
    # Pass through the linear backward pass
    dA_prev, dW, dG, dB = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, dG, dB


# TODO: implement batch norm for sigmoid and tanh and the rest of the code below    

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for the SIGMOID function.
    
    Arguments:
        dA: post-activation gradient, of any shape
        cache: 'Z' where we store for computing backward propagation efficiently
    
    Returns:
        dZ: Gradient of the cost with respect to Z
    """
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

def tanh_backward(dA, cache):
    """
    Implement the backward propagation for the TANH function.
    
    Arguments:
        dA: post-activation gradient, of any shape
        cache: 'Z' where we store for computing backward propagation efficiently
    
    Returns:
        dZ: Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = dA * (1 - np.tanh(Z)**2)
    
    return dZ

def linear_activation_backward(dA, cache, activation, lambda_reg=0.01):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
        dA: post-activation gradient for current layer l 
        cache: tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
        
    Returns:
        dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW: Gradient of the cost with respect to W (current layer l), same shape as W
        db: Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambda_reg)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambda_reg)
        
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambda_reg)
    elif activation == "linear":
        dZ = dA
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambda_reg)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    return dA_prev, dW, db

# without regularization
def custom_model_backward(AL, Y, caches, activations, last_activation="sigmoid", lambda_reg=0.01):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
        AL: probability vector, output of the forward propagation (L_model_forward() or custom_model_forward())
        Y: true "label" vector
        caches: list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        activations: list of activations for each layer (excluding input layer)
        last_activation: activation function for the last layer (default: "sigmoid" or "linear")
    
    Returns:
        grads: A dictionary with the gradients
    """
    grads = {}
    L = len(caches) # the number of layers    
    Y = Y.reshape(AL.shape) # Y is the same shape as AL

    if last_activation == "sigmoid":
        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    elif last_activation == "linear":
        dAL = AL - Y
    else:
        raise ValueError(f"Unsupported last activation: {last_activation}")
    
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, last_activation, lambda_reg)
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activations[l], lambda_reg)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads
  

def forward_and_backward_propagation(X, Y, parameters, activations, learning_rate=0.0075, num_classes=1, last_activation="sigmoid", lambda_reg=0.01):
    """
    Running forward and backward propagation on a single batch of data
    Returns cost, grads, parameters
    """    
    A, caches = custom_model_forward(X, parameters, activations, num_classes, apply_sigmoid=(last_activation == "sigmoid"))
    cost = BCE_WithLogitsLoss(A, Y, parameters, from_logits=(last_activation == "linear"), lambda_reg=lambda_reg)
    grads = custom_model_backward(A, Y, caches, activations, last_activation, lambda_reg)
    parameters = update_parameters(parameters, grads, learning_rate)
    
    return cost, grads, parameters
   



def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
        parameters: dictionary containing your parameters 
        grads: dictionary containing your gradients, output of L_model_backward
    
    Returns:
        parameters: dictionary containing your updated parameters 
    """
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters