"""
Using a deep convolutional neural network wit reduction factor of 32 to take a preprocessed image (680, 680, 3) as input and create
an encoding of (19, 19, 5, 85) as output for YOLO model.
"""
from src.torch_train.cnn.base_trainer import CNN_Model_Trainer
from src.torch_train.dataset import YoLoDataSet
from ..architectures.all_models import PreYoloCNN32

from torch import nn
from pathlib import Path
from torch import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import cv2

class PreprocessYOLO(CNN_Model_Trainer):
    def __init__(self, save_path, learning_rate=0.001, batch_size=16, epochs=10, is_gpu_train=True):
        super().__init__(save_path, is_gpu_train)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # apply image transformation for 64x64 input (train)
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: cv2.resize(x, (64, 64))),
                transforms.Lambda(lambda x: x.transpose(2, 0, 1) / 255.0),  # HWC -> CHW
                transforms.Lambda(lambda x: torch.FloatTensor(x)),
            ]
        )
        # test transform
        self.test_transform = transforms.Compose([
            transforms.Lambda(lambda x: cv2.resize(x, (64, 64))),
            transforms.Lambda(lambda x: x.transpose(2, 0, 1) / 255.0),  # HWC -> CHW
            transforms.Lambda(lambda x: torch.FloatTensor(x)),
        ])

    def load_data(self, only_test=False, dataset_path=None):
        """
        Load the data and set the trainloader, testloader, and classes.      
        """
        # load the dataset
        if dataset_path is None:
            dataset_path = "assets/produce_dataset/LVIS_Fruits_And_Vegetables" # tech debt: should be passed in from EnvironmentConfig
        
        dataset_path = Path(dataset_path)
        image_dir = dataset_path / "images"
        annotation_dir = dataset_path / "labels"

        if only_test:
            # load only test dataset
            test_dataset = YoLoDataSet(image_dir, annotation_dir, transform=self.test_transform)
            self.testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0) # num_workers=0 for CPU, will tune later

        else:
            # load train and test dataset
            train_dataset = YoLoDataSet(image_dir / "train", annotation_dir / "train", transform=self.transform)
            # using train and validation datasets provided
            self.trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0) # num_workers=0 for CPU, will tune later
            val_dataset = YoLoDataSet(image_dir / "val", annotation_dir / "val", transform=self.transform)
            self.testloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0) # num_workers=0 for CPU, will tune later

    def load_model(self):
        print(f"loading the model to {self.device}")
        self.cnn_model = PreYoloCNN32().to(self.device) # the pre-process model with reduction factor of 32 
        return self.cnn_model
        
    def train_model(self):
        """ Train the model """
        # validation of the data loaders, if not loaded, load the data
        if self.trainloader is None or self.testloader is None:
            self.load_data()
        
        # validation of the model, if not loaded, load the model
        if self.cnn_model is None:
            self.load_model()
        
        # loss function for YoLo encoding part to (19, 19, 3, 68)
        criterion = nn.MSELoss() # using Mean Squared Error loss function
        optimizer = optim.Adam(self.cnn_model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # decay learning rate by a factor of 0.1 every 7 epochs

        print(f"Training the model with {len(self.trainloader.dataset)} samples in batches of size {self.batch_size} on {self.device}")
        
        # training looop 
        for epoch in range(self.epochs):
            self.cnn_model.train()
            running_loss = 0.0

            for batch_idx, (images, targets) in enumerate(self.trainloader):
                # make sure data is on the same device 
                images = images.to(self.device)
                targets = targets.to(self.device) # should be (batch_size, 19, 19, 3, 68)

                # forward pass
                outputs = self.cnn_model(images) # should be (batch_size, 19, 19, 3, 68)

                # calculate loss
                loss = criterion(outputs, targets)  

                # back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if batch_idx % 10 == 9:    # print every 10 mini-batches
                    print(f"[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 10:.3f}")
        
            scheduler.step()
            # average loss for the epoch
            epoch_loss = running_loss / len(self.trainloader)
            print(f"Epoch {epoch + 1} completed, loss: {epoch_loss:.3f}")
            # save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.save_path.replace(".pth", f"_epoch_{epoch + 1}.pth")
                torch.save(self.cnn_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        # finally save the final model
        self.abstract_save_model()
        print(f"Saved final model to {self.save_path}. Training completed")
        
            