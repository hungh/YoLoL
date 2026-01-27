"""
Backward propagation for batch normalization
"""
import numpy as np

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

    # convert back to the original Z using Z_norm, var, and epsilon
    Z = Z_norm * np.sqrt(var + epsilon) + mean
    
    # Gradient of variance
    dvar = np.sum(dZ_norm * (Z - mean) * -0.5 * (var + epsilon)**(-1.5), axis=1, keepdims=True)
    
    # Gradient of mean
    dmean = np.sum(dZ_norm * -1 / np.sqrt(var + epsilon), axis=1, keepdims=True) + \
            dvar * np.sum(-2 * (Z - mean), axis=1, keepdims=True) / m
    
    # Gradient of Z (before batch norm)
    dZ_prev = (dZ_norm / np.sqrt(var + epsilon)) + \
              (dvar * 2 * (Z - mean) / m) + \
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
    Z = cache
    dZ = np.array(dA, copy=True)  # Just converting dA to a correct object
    
    # When z <= 0, set dz to 0
    dZ[Z <= 0] = 0    
    
    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for the SIGMOID function.
    
    Arguments:
        dA: post-activation gradient, of any shape
        cache: tuple containing (Z, linear_cache) where:
            - Z is the input to the activation function
            - linear_cache is the cache from the linear_forward pass
              (A_prev, W, G, B, Z_norm, mean, var, epsilon)
    
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
        cache: tuple containing (Z, linear_cache) where:
            - Z is the input to the activation function
            - linear_cache is the cache from the linear_forward pass
              (A_prev, W, G, B, Z_norm, mean, var, epsilon)
    
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
        dG: Gradient of the cost with respect to the scale parameter gamma
        dB: Gradient of the cost with respect to the shift parameter beta
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        # Pass through the linear backward pass
        dA_prev, dW, dG, dB = linear_backward(dZ, linear_cache, lambda_reg)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)        
        # Pass through the linear backward pass
        dA_prev, dW, dG, dB = linear_backward(dZ, linear_cache, lambda_reg)
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        # Pass through the linear backward pass
        dA_prev, dW, dG, dB = linear_backward(dZ, linear_cache, lambda_reg)
    elif activation == "linear":
        dA_prev, dW, dG, dB = linear_backward(dA, linear_cache, lambda_reg)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    return dA_prev, dW, dG, dB

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
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["dG" + str(L)], grads["dB" + str(L)] = linear_activation_backward(dAL, current_cache, last_activation, lambda_reg)
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, dG_temp, dB_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activations[l], lambda_reg)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["dG" + str(l + 1)] = dG_temp
        grads["dB" + str(l + 1)] = dB_temp
    
    return grads
  
