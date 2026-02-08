from . import model_1_train
from .mobile import mobilenetv2_trainer
from pathlib import Path

MODEL_DICT = {
    "SM_2C1M3FC": model_1_train.SAM_Model_1,
    "MOBILENETV2": mobilenetv2_trainer.MobileNetV2_Trainer
}

if __name__ == "__main__":
    # parsing input arguments for cnn model name
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m src.torch_train.cnn <model_name> <epochs>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    epochs = int(sys.argv[2])    
    
    predict = False
    if len(sys.argv) == 4:
        predict = sys.argv[3]

    
    print(f'Training {model_name} model...')
    cur_dir = Path.cwd()
    trainer = MODEL_DICT[model_name](save_path=f"{cur_dir}/saved_models/{model_name}.pt", epochs=epochs)
    trainer.train_model()

    if predict:
        trainer.predict("./assets/mnist/signs/my_image.jpg")

"""Usage:
python -m src.torch_train.cnn SM_2C1M3FC 10 False
python -m src.torch_train.cnn MOBILENETV2 10 True
"""