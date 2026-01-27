"""
Metrics for mini-batch training.
"""
import numpy as np
from sklearn.metrics import hamming_loss, f1_score

def multi_label_metrics(Y_pred, Y_true, prediction_threshold: float = 0.5):
    """
    Calculate multi-label metrics for binary predictions.
    
    Parameters:
    -----------
    Y_pred : numpy.ndarray
        Predicted probabilities of shape (n_classes, n_samples)
    Y_true : numpy.ndarray
        True labels of shape (n_classes, n_samples)
    prediction_threshold : float, optional
        Threshold for converting probabilities to binary predictions (default: 0.5)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - exact_match: fraction of samples with all labels correct
        - hamming_loss: fraction of wrong labels
        - f1_score: micro-averaged F1 score
        - avg_predicted: average predicted probability
        - avg_actual: average actual probability
    """
    
    # Convert to binary predictions
    Y_pred_binary = (Y_pred > prediction_threshold).astype(int)
    
    # Exact match ratio (strict accuracy)
    exact_match = np.mean(np.all(Y_pred_binary == Y_true, axis=0))
    
    # Hamming loss (fraction of wrong labels)
    hamming = hamming_loss(Y_true, Y_pred_binary)
    
    # F1 score (micro-averaged)
    f1 = f1_score(Y_true, Y_pred_binary, average='micro')
    
    return {
        'exact_match': exact_match,
        'hamming_loss': hamming,
        'f1_score': f1,
        'avg_predicted': np.mean(Y_pred_binary),
        'avg_actual': np.mean(Y_true)
    }