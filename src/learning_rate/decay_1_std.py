"""
Learning rate decay implementation. Standard decay.

"""

def decay_1_std(epoch, initial_lr=0.001, decay_rate=0.1):
    """
    Decay learning rate by a factor of decay_rate every 10 epochs
    
    Args:
        epoch: Current epoch number
        initial_lr: Initial learning rate
        decay_rate: Decay rate (e.g., 0.1 means 10x reduction)
    
    Returns:
        decayed_lr: Learning rate for current epoch
    """
    return initial_lr * (decay_rate ** (epoch // 10))
