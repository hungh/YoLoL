import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN with 2 conv layers, 1 max pool layer, and 3 fully connected layers
class CNN_Small_2C1M3FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PreYoloCNN32(nn.Module):
    """
    A deep convolutional neural network wit 'reduction factor of 32' to take a preprocessed image (64, 64, 3) as input and create
    an encoding of (19, 19, 3, 68) as output for YOLO model
    the encoding has height and width of 19, 3 anchor boxes and 68 classes for each anchor box
    NOTE: since the preprocessing model is small, we are actually not using the reduction factor of 32 for now.
    We will update this class or add another class to use the reduction factor of 32. @hungd to come back
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=2, stride=2, padding=5)   # 37x37x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)   # 19x19x64
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)   # just increasing channels
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)  # 17x17x256
        self.bn4 = nn.BatchNorm2d(256)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((19, 19))
        self.final = nn.Conv2d(in_channels=256, out_channels=3 * 68, kernel_size=1, stride=1, padding=0)  # desire output (19, 19, 204)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.adaptive_pool(x)
        x = self.final(x)
        return x.view(-1, 19, 19, 3, 68) # 3 anchor boxes, 68 classes
       