import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from .. base_trainer import CNN_Model_Trainer
from src.configs import EnvironmentConfig
from ...dataset.ASLDataSet import ASLDataSet
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path


# this class loads a pre-trained MobileNetV2 model and fine-tunes it for a new task, sign language recognition
class MobileNetV2_Trainer(CNN_Model_Trainer):
    def __init__(self, save_path, epochs=20):
        """
        Args:
            save_path (str): path to save the model
            epochs (int, optional): number of epochs to train. Defaults to 20.
        """
        super().__init__(save_path)
        self.env_config = EnvironmentConfig()
        self.epochs = epochs
        self.classes = 24
        self.learning_rate = 0.0001   
       
    
    def load_data(self, only_test=False):
        # load the American Sign Language dataset
        dataset_dir = self.env_config.get_sign_path()
        train_path = os.path.join(dataset_dir, 'sign_mnist_train.csv')
        test_path = os.path.join(dataset_dir, 'sign_mnist_test.csv')

        if not only_test:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),                
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Grayscale(num_output_channels=3), # to RGB channel data
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            train_dataset = ASLDataSet(train_path, transform=train_transform)
            # get the number of classes from the dataset
            assert train_dataset.num_classes == self.classes, f"Number of classes in the dataset is {train_dataset.num_classes}, expected {self.classes}"
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
            self.trainloader = train_loader
            print(f"Training data size: {len(train_dataset)}")

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3), # to RGB channel data
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_dataset  = ASLDataSet(test_path, transform=test_transform)
        test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
        self.testloader = test_loader

        print(f"Test data size: {len(test_dataset)}")

    
    def get_base_model(self):
        # Load pretrained MobileNetV2
        base_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
    
        # Replace the final classifier layer
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, self.classes)  # MobileNetV2 has 1280 in_features
        )
        
        return base_model

    def train_model(self):

        # check if the model exists on disk
        saved_model_dir = self.env_config.get_saved_model_dir()
        if Path(self.save_path).exists():
            print(f"Loading model from {saved_model_dir}")
            self.load_data(only_test=True)
            # creating a model architecture
            base_model = self.get_base_model()
            
            # load weights from disk
            base_model.load_state_dict(torch.load(self.save_path, weights_only=True))
            self.cnn_model = base_model.to(self.device)
            self.cnn_model.eval() # set model to evaluation mode (inference mode)
        else:
            self.load_data()
            print(f"Starting training model: {self}")
            based_model = self.get_base_model()
            self.cnn_model = based_model.to(self.device)

            # Freeze all convolutional layers (features)
            for param in self.cnn_model.features.parameters():
                param.requires_grad = False
            
            # Unfreeze the classifier FC layers
            for param in self.cnn_model.classifier.parameters():
                param.requires_grad = True

            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            
            # Only optimize the trainable parameters (classifier)
            optimizer = optim.Adam(self.cnn_model.classifier.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            # Training loop
            self.cnn_model.train()
            for epoch in range(self.epochs):
                running_loss = 0.0
                for inputs, labels in self.trainloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.cnn_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()                    
                    
                    running_loss += loss.item()

                scheduler.step()
                print(f'Epoch {epoch+1}/{self.epochs}, Loss: {running_loss/len(self.trainloader):.4f}')
            
            # Save model
            self.abstract_save_model()
        # end training
        print("Training completed")
        # run test
        self.test_model()
        

    def abstract_save_model(self):
        torch.save(self.cnn_model.state_dict(), self.save_path)

    def test_model(self):
        if self.cnn_model is None:
            raise ValueError("MobileNetV2 model is not loaded")
        
        self.cnn_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.cnn_model(images)
                preds = outputs.argmax(dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        # Convert lists of tensors to single tensors
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # calculate metrics
        accuracy = (all_preds == all_labels).float().mean()
        print(f'Accuracy: {accuracy:.4f}')

        cm = confusion_matrix(all_labels, all_preds)

        sns.heatmap(cm, annot=True, fmt="d")
        plt.show()
   

    def predict(self, image_path):
        input_image = Image.open(image_path)
        # show image
        plt.imshow(input_image, cmap='gray')
        plt.title('Input Image')
        plt.show()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        if self.cnn_model is None:
            raise ValueError("MobileNetV2 model is not loaded")

        with torch.no_grad():
            prediction = self.cnn_model(input_batch)
        
        probabilities = torch.nn.functional.softmax(prediction[0], dim=0)
        predicted_classes = torch.argmax(probabilities).item() # index of the predicted class
        confidence = probabilities[predicted_classes].item()

        print(f"Predicted class: {predicted_classes}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: {probabilities}")

    def __str__(self):
        return f"MobileNetV2_Trainer(epochs={self.epochs}, classes={self.classes}, learning_rate={self.learning_rate})"
        
        