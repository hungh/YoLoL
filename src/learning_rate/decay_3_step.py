"""
Learning rate decay implementation. Step decay.
"""

def decay_3_step(epoch, initial_lr=0.001, decay_rate=0.5, step_size=10):
    """
    Step decay of learning rate
    
    Args:
        epoch: Current epoch number
        initial_lr: Initial learning rate
        decay_rate: Decay rate (e.g., 0.5 means 50% reduction)
        step_size: Number of epochs before decay
    
    Returns:
        decayed_lr: Learning rate for current epoch
    """
    return initial_lr * (decay_rate ** (epoch // step_size))
