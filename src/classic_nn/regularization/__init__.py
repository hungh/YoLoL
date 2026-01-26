"""
Regularization for the classic nn
"""

import numpy as np

def L2_regularization(m, parameters, lambd = None, use_batch_norm = False):
    """
    L2 regularization for the classic nn. If lambd is None, return 0 
    Input:
        m: the number of examples
        parameters: the parameters of the model
        lambd: the lambda value for the regularization
        use_batch_norm: whether to use batch normalization
    Output:
        L2_cost: the L2 cost
    """
    if lambd is None:
        return 0  
    L2_cost = 0
    L = len(parameters) // (3 if use_batch_norm else 2)
    for l in range(L):
        L2_cost += np.sum(np.square(parameters['W' + str(l+1)]))
    return (lambd/(2 * m)) * L2_cost 