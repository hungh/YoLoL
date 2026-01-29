"""
Learning rate decay implementation. Discrete decay.
"""

def decay_4_discrete(epoch, initial_lr=0.001, decay_points=[5, 10, 15]):
    """
    Discrete decay of learning rate at specific epochs
    
    Args:
        epoch: Current epoch number
        initial_lr: Initial learning rate
        decay_points: List of epochs where learning rate should be reduced
    
    Returns:
        decayed_lr: Learning rate for current epoch
    """
    lr = initial_lr
    for point in decay_points:
        if epoch >= point:
            lr *= 0.5
    return lr
