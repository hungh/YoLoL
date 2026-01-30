"""
Utility functions for training CNN models
"""

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch

def imshow_with_unnormalize(img):
    """
    Show a single image. Unnormalize the image first.
    NOTE: the image was expected to be normalized to [-1, 1]
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_images_in_grid(loader, classes, batch_size = 4):
    """
    Show images in a grid.
    
    Args:
        loader: DataLoader
        classes: list of class names
        batch_size: batch size
    Returns:
        images: images
        labels: labels
    """
    # get some random training images
    dataiter = iter(loader)
    images, labels = next(dataiter)

    # show images
    imshow_with_unnormalize(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    return images, labels


def predict_image(model, image):
    """
    Predict the class of an image
    Args:
        model: the model to predict
        image: the image to predict
    Returns:
        predicted: the predicted class
    """
    _, predicted = torch.max(model(image).data, 1)
    return predicted.cpu()