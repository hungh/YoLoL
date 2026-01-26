from . import train_mlayer_mini_batch
from . import bn_tiny_model
from . import tiny_model
from . import bn_opt_mini_batch_model
if __name__ == "__main__":
    # train_mlayer_mini_batch(print_cost=True)
    # bn_tiny_model.train_model()
    # tiny_model.train_model()
    bn_opt_mini_batch_model.train_model()

"""Usage:
python -m src.classic_train
"""