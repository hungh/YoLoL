"""
Classic neural network implementations for comparison with YOLO models.
"""
import numpy as np
from . import optimizers
from .utils import get_num_layers
from .math import sigmoid, tanh, relu
from .loss import BCE_WithLogitsLoss


def initialize_parameters_deep(layer_dims: list, optimizer_instance: optimizers.OptimizerFactory=None) -> dict:
    """
    Initialize parameters for deep neural network.
    
    Args:
        layer_dims: list of integers representing the number of units in each layer
    
    Returns:
        Dictionary containing initialized parameters
        W1, b1, W2, b2, ...
    """    
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 1/np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    
    assert(parameters['W' + str(1)].shape == (layer_dims[1], layer_dims[0]))
    assert(parameters['b' + str(1)].shape == (layer_dims[1], 1))
    
    if optimizer_instance is not None:
        optimizer_instance.initialize_parameters(parameters, layer_dims)
    
    return parameters


    
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    
    Arguments:
        A: activations from previous layer (or input data)
        W: weights matrix
        b: bias vector
    
    Returns:
        Z: the input of the activation function
        cache: a python tuple containing "linear_cache" and "activation_cache"
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer
    
    Arguments:
        A_prev: activations from previous layer (or input data)
        W: weights matrix
        b: bias vector
        activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
        A: the output of the activation function
        cache: a python tuple containing "linear_cache" and "activation_cache"
    """
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == 'tanh':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
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
    L = get_num_layers(parameters)
    
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
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
    L = get_num_layers(parameters)

    # check layer_names length
    if len(layer_names) - 1 != L:
        raise ValueError(f"Layer names length {len(layer_names)} does not match parameters length {L}")
    
    for l in range(1, L + 1):
        A_prev = A

        activation = layer_names[l -1]
        if activation not in ['relu', 'tanh', 'sigmoid']:
            raise ValueError(f"Unsupported activation: {activation}")

        current_activation = 'sigmoid' if (l == L and apply_sigmoid) else 'relu'

        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], current_activation)
        caches.append(cache)
        
    if num_classes is not None and A.shape[0] != num_classes:
        raise ValueError(f"Output layer size {A.shape[0]} does not match num_classes {num_classes}")
    
    return A, caches



def linear_backward(dZ, cache, lambda_reg=None):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
        dZ: Gradient of the cost with respect to the linear output (of current layer l)
        cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
        dA_prev: Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW: Gradient of the cost with respect to W (current layer l), same shape as W
        db: Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, _ = cache
    m = A_prev.shape[1]
    
    regularization_weight = 0
    if lambda_reg is not None:
        regularization_weight = lambda_reg / m
    
    dW = 1./m * np.dot(dZ, A_prev.T) + regularization_weight * W
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def relu_backward(dA, cache):
    """
    Implement the backward propagation for the RELU function.
    
    Arguments:
        dA: post-activation gradient, of any shape
        cache: 'Z' where we store for computing backward propagation efficiently
    
    Returns:
        dZ: Gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    return dZ

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
  

def gradient_descent(X, Y, parameters, activations, learning_rate=0.0075, num_classes=1, last_activation="sigmoid", lambda_reg=0.01, optimizer_instance: optimizers.OptimizerFactory=None):
    """
    Running forward and backward propagation on a single batch of data
    Returns cost, grads, parameters
    """    
    A, caches = custom_model_forward(X, parameters, activations, num_classes, apply_sigmoid=(last_activation == "sigmoid"))
    cost = BCE_WithLogitsLoss(A, Y, parameters, from_logits=(last_activation == "linear"), lambda_reg=lambda_reg)
    grads = custom_model_backward(A, Y, caches, activations, last_activation, lambda_reg)
    parameters = update_parameters(parameters, grads, learning_rate, optimizer_instance)
    
    return cost, grads, parameters
   



def update_parameters(parameters, grads, learning_rate, optimizer_instance: optimizers.OptimizerFactory=None):
    """
    Update parameters using gradient descent
    
    Arguments:
        parameters: dictionary containing your parameters 
        grads: dictionary containing your gradients, output of L_model_backward
    
    Returns:
        parameters: dictionary containing your updated parameters 
    """
    L = get_num_layers(parameters) # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        if optimizer_instance is not None:
            optimizer_instance.update_parameters_once(
                parameters, 
                grads,
                l + 1
            )
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["vW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["vb" + str(l+1)]
            
        else:
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters