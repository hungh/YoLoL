"""
Evaluation utilities for neural networks
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ...classic_nn.batch_norm.forward import custom_model_forward


def predict(X, parameters, activations, num_classes):
    """
    Predict the class labels for given input data.
    
    Args:
        X: Input data of shape (n_features, n_samples)
        parameters: Dictionary containing model parameters
        activations: List of activation functions for each layer
        num_classes: Number of classes (1 for binary, >1 for multi-class)
    
    Returns:
        Predicted class labels
    """
    A, _ = custom_model_forward(X, parameters, activations, num_classes, apply_sigmoid=(num_classes == 1))
    return A

def evaluate_model(X, Y, parameters, activations, num_classes=1, last_activation="sigmoid", prediction_threshold=0.5, adjust_prediction_func=None):
    """
    Evaluate the model on given data.
    
    Args:
        X: Input data of shape (n_features, n_samples)
        Y: True labels of shape (1, n_samples)
        parameters: Dictionary containing model parameters
        activations: List of activation functions for each layer
        num_classes: Number of classes (1 for binary, >1 for multi-class)
        last_activation: Activation function for the last layer ("sigmoid" or "softmax")
    
    Returns:
        None (prints evaluation metrics)
    """
    A, _ = custom_model_forward(X, parameters, activations, num_classes, apply_sigmoid=(last_activation == "sigmoid"))

    print(f"A shape: {A.shape}")
    print(f"Y shape: {Y.shape}")
    
    # Reshape Y from (1, n_samples) to (n_samples,)
    Y_flat = Y.reshape(-1)
    A_flat = A.reshape(-1)
    
    if adjust_prediction_func is not None:
        A_flat = adjust_prediction_func(A_flat, last_activation)

    # calculate accuracy
    accuracy = np.mean((A_flat > prediction_threshold) == Y_flat)
    print("Metrics:")   

    # calculate f1 score    
    accuracy = accuracy_score(Y_flat, A_flat > prediction_threshold)
    precision = precision_score(Y_flat, A_flat > prediction_threshold)
    recall = recall_score(Y_flat, A_flat > prediction_threshold)
    f1 = f1_score(Y_flat, A_flat > prediction_threshold)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
