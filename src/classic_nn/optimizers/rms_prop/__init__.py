"""
RMS Prop optimizer implementation
"""
import numpy as np

class RMSProp:
    def __init__(self, beta: float):
        self.beta = beta
        self.epsilon = 1e-8
    
    def update_parameters_once(self, parameters, grads, layer_index: int, learning_rate: float):
        """
        Update parameters using RMS Prop. 
        """
        if "_batch_norm" in parameters:
            parameters["sW" + str(layer_index)] = self.beta * parameters["sW" + str(layer_index)] + (1 - self.beta) * grads["dW" + str(layer_index)] ** 2
            parameters["sG" + str(layer_index)] = self.beta * parameters["sG" + str(layer_index)] + (1 - self.beta) * grads["dG" + str(layer_index)] ** 2
            parameters["sB" + str(layer_index)] = self.beta * parameters["sB" + str(layer_index)] + (1 - self.beta) * grads["dB" + str(layer_index)] ** 2
            parameters["G" + str(layer_index)] = parameters["G" + str(layer_index)] - learning_rate * grads["dG" + str(layer_index)] / (np.sqrt(parameters["sG" + str(layer_index)] + self.epsilon) + self.epsilon)
            parameters["B" + str(layer_index)] = parameters["B" + str(layer_index)] - learning_rate * grads["dB" + str(layer_index)] / (np.sqrt(parameters["sB" + str(layer_index)] + self.epsilon) + self.epsilon)
        else:
            parameters["sW" + str(layer_index)] = self.beta * parameters["sW" + str(layer_index)] + (1 - self.beta) * grads["dW" + str(layer_index)] ** 2
            parameters["sb" + str(layer_index)] = self.beta * parameters["sb" + str(layer_index)] + (1 - self.beta) * grads["db" + str(layer_index)] ** 2
            parameters["b" + str(layer_index)] = parameters["b" + str(layer_index)] - learning_rate * grads["db" + str(layer_index)] / (np.sqrt(parameters["sb" + str(layer_index)] + self.epsilon) + self.epsilon)

        parameters["W" + str(layer_index)] = parameters["W" + str(layer_index)] - learning_rate * grads["dW" + str(layer_index)] / (np.sqrt(parameters["sW" + str(layer_index)] + self.epsilon) + self.epsilon)
        
    def initialize_parameters(self, parameters, layer_dims):
        if '_batch_norm' in parameters:
            for l in range(1, len(layer_dims)):
                parameters["sW" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
                parameters["sG" + str(l)] = np.zeros((layer_dims[l], 1))
                parameters["sB" + str(l)] = np.zeros((layer_dims[l], 1))
        else:
            for l in range(1, len(layer_dims)):
                parameters["sW" + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
                parameters["sb" + str(l)] = np.zeros((layer_dims[l], 1))
