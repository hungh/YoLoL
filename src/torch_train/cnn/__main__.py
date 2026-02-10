from . import model_1_train
from .mobile import mobilenetv2_trainer
from .yolo import preprocess_yolo, pre_trained_yolo
from pathlib import Path
import sys

MODEL_DICT = {
    "SM_2C1M3FC": model_1_train.SAM_Model_1,
    "MOBILENETV2": mobilenetv2_trainer.MobileNetV2_Trainer,
    "PREPROCESS_YOLO": preprocess_yolo.PreprocessYOLO,
    "PRETRAINED_YOLO": pre_trained_yolo.PreTrainedYOLO
}

if __name__ == "__main__":
    # if there is no argument run the model without any parameters
    
    model_name = sys.argv[1]
    if model_name == "PRETRAINED_YOLO":    
        trainer = MODEL_DICT[model_name]()
        if len(sys.argv) == 3:
            image_path = sys.argv[2]    
        else:
            image_path = "assets/produce_dataset/LVIS_Fruits_And_Vegetables/images/val/val/000000555239.jpg"
        trainer.predict(image_path)
        sys.exit(0)
      
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
python -m src.torch_train.cnn PREPROCESS_YOLO 10 False
python -m src.torch_train.cnn PRETRAINED_YOLO <image_path>
"""