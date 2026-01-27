"""
Regularization for the classic nn (neural network)
"""

import numpy as np
from ..utils import get_num_layers

def L2_regularization(m, parameters, lambd = None):
    """
    L2 regularization for the classic nn. If lambd is None, return 0 
    Input:
        m: the number of examples
        parameters: the parameters of the model
        lambd: the lambda value for the regularization        
    Output:
        L2_cost: the L2 cost
    """
    if lambd is None:
        return 0  
    L2_cost = 0
    L = get_num_layers(parameters)
    for l in range(L):
        L2_cost += np.sum(np.square(parameters['W' + str(l+1)]))
    return (lambd/(2 * m)) * L2_cost 