from . import model_1_train
from pathlib import Path

MODEL_DICT = {
    "SM_2C1M3FC": model_1_train.SAM_Model_1,
}

if __name__ == "__main__":
    # parsing input arguments for cnn model name
    import sys
    model_name = sys.argv[1]
    
    print(f'Training {model_name} model...')
    cur_dir = Path.cwd()
    MODEL_DICT[model_name](save_path=f"{cur_dir}/saved_models/{model_name}.pt").train_model()

"""Usage:
python -m src.torch_train.cnn SM_2C1M3FC
"""