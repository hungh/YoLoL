"""
Data utilities for neural networks
"""
from sklearn.datasets import make_moons
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import numpy as np

def generate_binary_classification_data():
    X, Y = make_moons(n_samples=15000, noise=0.2, random_state=42)
    return X, Y


def generate_multilabel_dataset(n_samples=1000, n_features=32*32*3, n_classes=3, avg_labels=3, test_size=0.2, random_state=42, split_train_test=True):
    """
    Generate a tiny multi-label dataset for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features per sample
    n_classes : int
        Number of classes
    avg_labels : int
        Average number of labels per sample
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, Y_train, X_test, Y_test : tuple
        Training and test datasets
    """    
    # avg_labels    # Average number of objects per image

    # Generate the data
    X, Y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=avg_labels,  # This is the "Poisson" mean for label count
        allow_unlabeled=False, # Ensures every 'image' has at least one object
        random_state=random_state
    )

    print(f"Data set has been initialized with {n_samples} samples, {n_features} features, {n_classes} classes, {avg_labels} average labels per sample")
    # split the dataset into train and test
    if split_train_test:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, shuffle=True)
        
        print("Train label distribution:", np.mean(Y_train, axis=0))
        print("Test label distribution:", np.mean(Y_test, axis=0))
        # Returns X as (Features, Samples) and Y as (Classes, Samples)    
        return X_train.T, Y_train.T, X_test.T, Y_test.T 
    else:
        X_train, Y_train = X, Y
        X_test, Y_test = None, None

        return X_train.T, Y_train.T, X_test, Y_test

    
