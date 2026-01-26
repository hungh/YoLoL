"""
Utility functions for neural networks
"""

def get_num_layers(parameters: dict) -> int:
    """
    Get the number of layers from parameters dictionary.
    
    Args:
        parameters: dictionary containing parameters
        
    Returns:
        Number of layers
    """
    if '_num_layers' in parameters:
        return parameters['_num_layers']

    layers = len([k for k in parameters.keys() if k.startswith('W')])
    parameters['_num_layers'] = layers
    return layers
