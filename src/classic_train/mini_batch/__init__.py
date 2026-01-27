# src/classic_train/mini_batch/__init__.py
"""
Mini-batch training package
"""

# Import the main class for backward compatibility
from .trainer import MiniBatchTrainer

# Also expose at package level
__all__ = ['MiniBatchTrainer']