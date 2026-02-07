import torch
from .utils import show_images_in_grid, predict_image

# abstract class for CNN model trainer
class CNN_Model_Trainer:
    def __init__(self, save_path, is_gpu_train=True):
        self.save_path = save_path
        self.trainloader = None
        self.testloader = None
        self.classes = None
        self.cnn_model = None
        self.is_gpu_train = is_gpu_train
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.is_gpu_train else "cpu")

    def load_data(self, only_test=False):
        """
        Load the data and set the trainloader, testloader, and classes.      
        """
        raise NotImplementedError
    
    def train_model(self):
        """
        Train the model
        """
        raise NotImplementedError

    def test_model(self):
        print(f'is_gpu_train: {self.is_gpu_train}')
        print(f'device: {self.device}')

        # check if the model is loaded
        if self.cnn_model is None:
            self.cnn_model.load_state_dict(torch.load(self.save_path, weights_only=True))
            self.cnn_model = self.cnn_model.to(self.device)
        
        # load the test data
        if self.testloader is None:
            self.load_data(only_test=True)
    
        # print test images
        images, labels = show_images_in_grid(self.testloader, self.classes, batch_size = 4)
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # predict the images
        predicted = predict_image(self.cnn_model, images)
        predicted = predicted.cpu()
        
        # print the predicted classes
        print('Predicted: ', ' '.join(f'{predicted[j]} {self.classes[predicted[j]]:5s}' for j in range(4)))
    
    def abstract_save_model(self):
        """
        Save the model
        """
        if self.save_path is None or self.cnn_model is None:
            raise ValueError("Save path or CNN model is not specified")
        torch.save(self.cnn_model.state_dict(), self.save_path)