# src/classic_train/mini_batch_generic.py
"""
DEPRECATED: This file has been moved to src/classic_train/mini_batch/
Please use: from src.classic_train.mini_batch import MiniBatchTrainer

This file will be removed in a future version.
"""

import warnings
warnings.warn(
    "mini_batch_generic is deprecated. Use 'from src.classic_train.mini_batch import MiniBatchTrainer' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import for backward compatibility
from .mini_batch import MiniBatchTrainer

# Re-export everything
__all__ = ['MiniBatchTrainer']