"""
Implementation of Adam Optimizer
"""
class Adam:
    
    def __init__(self, beta1: float, beta2: float, epsilon: float):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


    def update_parameters_once(self, parameters, grads):
        pass
    
    def initialize_parameters(self, parameters, layer_dims):
        pass

