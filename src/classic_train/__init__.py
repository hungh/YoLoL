"""
Classic Neural Network Implementation using classic_nn package
"""
from .binary_model import train_model as binary_train_model
from .multi_label_model import train_model as multi_label_train_model

__all__ = [
    "binary_train_model",
    "multi_label_train_model"
]
