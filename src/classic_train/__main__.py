from . import multi_label_model
from . import binary_model
from . import batch_norm_model

if __name__ == "__main__":
    print('Training binary model...')
    binary_model.train_model()
    # multi_label_model.train_model()
    # batch_norm_model.train_model()

"""Usage:
python -m src.classic_train
"""