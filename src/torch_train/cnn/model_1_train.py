import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from .base_trainer import CNN_Model_Trainer
from . import CNN_Model_Factory


class SAM_Model_1(CNN_Model_Trainer):

    Y_HAT_CLASSES = [
        "plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
    ]
    
    def __init__(self, save_path):
        super().__init__(save_path)
    
    def load_data(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        batch_size = 4
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        self.classes = self.Y_HAT_CLASSES

    def train_model(self):

        self.load_data()
        cnn_model = CNN_Model_Factory.get_model("SM_2C1M3FC")()
        self.cnn_model = cnn_model.to(self.device)

        # define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.cnn_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

        self.abstract_save_model()