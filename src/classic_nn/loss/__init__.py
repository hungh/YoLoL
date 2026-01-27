"""
Loss functions for neural networks.
"""
from ..regularization import L2_regularization 
import numpy as np


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


