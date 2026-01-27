"""
Mini-batch utilities for training.
"""
def mini_batch_generator(X_train, Y_train, batch_size):
    """
    Generate batches of data for training. No validation on the inputs is performed.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features of shape (n_features, n_samples)
    Y_train : numpy.ndarray
        Training labels of shape (n_classes, n_samples)
    batch_size : int
        Size of each batch
        
    Yields:
    -------
    tuple
        Batch of features and labels
    """
    for i in range(0, X_train.shape[1], batch_size):
        yield X_train[:, i:i+batch_size], Y_train[:, i:i+batch_size]


