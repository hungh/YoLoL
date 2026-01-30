"""
This is the root package of the cnn module
"""
from .architectures.all_models import CNN_Small_2C1M3FC

# a dictionary of CNN models
CNN_MODELS = {
    "SM_2C1M3FC": CNN_Small_2C1M3FC
}



class CNN_Model_Factory:
    
    @staticmethod
    def get_model(model_name):
        return CNN_MODELS[model_name]

    
    @staticmethod
    def get_model_names():
        return list(CNN_MODELS.keys())
