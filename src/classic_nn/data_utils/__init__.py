"""
Data utilities for neural networks
"""
from sklearn.datasets import make_moons

def generate_binary_classification_data():
    X, Y = make_moons(n_samples=15000, noise=0.2, random_state=42)
    return X, Y