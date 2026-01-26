import numpy as np

class Momentum:
    def __init__(self, beta: float):
        self.beta = beta
    
    def update_parameters_once(self, parameters, grads, layer_index: int):
        """
        Update parameters using momentum. 
        """
        parameters["vW" + str(layer_index)] = self.beta * parameters["vW" + str(layer_index)] + (1 - self.beta) * grads["dW" + str(layer_index)]
        parameters["vb" + str(layer_index)] = self.beta * parameters["vb" + str(layer_index)] + (1 - self.beta) * grads["db" + str(layer_index)]

        
    def initialize_parameters(self, parameters, layer_dims):
        if '_batch_norm' in parameters:
            for l in range(1, len(layer_dims)):
                parameters["vW" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
                parameters["vG" + str(l)] = np.zeros((layer_dims[l], 1))
                parameters["vB" + str(l)] = np.zeros((layer_dims[l], 1))
        else:
            for l in range(1, len(layer_dims)):
                parameters["vW" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
                parameters["vb" + str(l)] = np.zeros((layer_dims[l], 1))