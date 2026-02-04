from . import ResNet50_trainer

MODEL_DICT = {
    "resnet50": ResNet50_trainer
}

if __name__ == "__main__":
    import sys
    import os
    # os.environ['TF_METAL_DEVICE_NAME'] = '0' # use GPU, uncomment to use GPU, check ResNet50.py for tips how to enable GPU
    model_name = sys.argv[1]
    epochs = int(sys.argv[2])
    print(f"Training {model_name} for {epochs} epochs")
    MODEL_DICT[model_name].train_model(force_train=False, epochs=epochs)

"""
Usage:
python -m src.tf_train.resnet resnet50 5
"""