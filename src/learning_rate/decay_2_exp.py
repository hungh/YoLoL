"""
Exponential decay implementation.

"""

def decay_2_exp(epoch, initial_lr=0.001, decay_rate=0.96):
    """
    Exponential decay implementation.
    
    Args:
        epoch: Current epoch number
        initial_lr: Initial learning rate
        decay_rate: Decay rate (e.g., 0.96 means 4% reduction per epoch)
    
    Returns:
        decayed_lr: Learning rate for current epoch
    """
    return initial_lr * (decay_rate ** epoch)
