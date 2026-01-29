"""
Learning rate decay implementations.
Expose all decay functions.
"""

from .decay_1_std import decay_1_std
from .decay_2_exp import decay_2_exp
from .decay_3_step import decay_3_step
from .decay_4_discrete import decay_4_discrete

__all__ = [
    "decay_1_std",
    "decay_2_exp",
    "decay_3_step",
    "decay_4_discrete",
]

class LearningRateDecay:
    @staticmethod
    def get_decay_function(decay_type):
        if decay_type == "std":
            return decay_1_std
        elif decay_type == "exp":
            return decay_2_exp
        elif decay_type == "step":
            return decay_3_step
        elif decay_type == "discrete":
            return decay_4_discrete
        else:
            raise ValueError(f"Unknown decay type: {decay_type}")

