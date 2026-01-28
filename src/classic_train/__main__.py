from . import multi_label_model
from . import binary_model
from . import batch_norm_model
from . import binary_model_adam
from . import multi_label_model_adam

MODEL_DICT = {
    "binary": binary_model,
    "multi_label": multi_label_model,
    "batch_norm": batch_norm_model,
    "binary_adam": binary_model_adam,
    "multi_label_adam": multi_label_model_adam    
}

if __name__ == "__main__":
    # parsing input arguments for model name
    import sys
    model_name = sys.argv[1]
    
    print(f'Training {model_name} model...')
    MODEL_DICT[model_name].train_model()

"""Usage:
python -m src.classic_train binary
"""