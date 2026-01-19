"""
Classic Neural Network Implementation using classic_nn package
"""
import time
import sys
import os
from pathlib import Path
from src.load.read_produce_dataset import load_yaml_classes
from .classic_nn import *


def custom_layer_model(X, Y, layers_dims, activations, learning_rate=0.0075, num_iterations=3000, print_cost=False, last_activation="sigmoid"):
    """
    Implements a multi-layer neural network
    
    Arguments:
        X: input data of shape (n_x, m)
        Y: true "label" vector of shape (1, m)
        layers_dims: list containing the input size and each layer size
        activations: list containing the activation function for each layer
        learning_rate: learning rate of the gradient descent update rule
        num_iterations: number of iterations of the optimization loop
        print_cost: if True, it prints the cost every 100 steps
        last_activation: activation function for the last layer (default: "sigmoid")
    
    Returns:
        parameters: parameters learnt by the model
    """
    costs = []
    
    parameters = initialize_parameters(layers_dims)
    
    for i in range(num_iterations):
        A, caches = custom_model_forward(X, parameters, activations, last_activation)
        cost = BCE_WithLogitsLoss(A, Y, from_logits=(last_activation == "linear"))
        costs.append(cost)
        grads = custom_model_backward(A, Y, caches, activations, last_activation)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    
    return parameters

# train the model using YoloL data
def train_model():    
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.load.read_produce_dataset import process_dataset, scale_data
    print("Loading dataset...")

    start_time = time.time()

    project_root = os.getcwd()
    images_root = os.path.join(project_root, "assets/produce_dataset/LVIS_Fruits_And_Vegetables")
    yaml_path = os.path.join(project_root, "assets/produce_dataset/LVIS_Fruits_And_Vegetables/data.yaml")

    data = load_yaml_classes(yaml_path)

    train_images = os.path.join(images_root, data["train"])
    train_labels = train_images.replace("images", "labels")

    # TODO: need to write a function to process X, Y in batches instead as memory intensive
    X, Y = process_dataset(train_images, train_labels, target_size=64)
    X = scale_data(X, method='minmax')  # Scale to [0, 1]    

    # Split data into train/validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    layers_dims = [X.shape[0], 2048, 1024, 512, 256, 128, Y.shape[0]]
    activations = ["relu"] * (len(layers_dims) - 1) + ["linear"]
    hyperparameters = {
        "learning_rate": 0.0075,
        "num_iterations": 3000,
        "batch_size": 1024,
        "print_cost": True
    }
    
    model_parameters = custom_layer_model(X_train, Y_train, layers_dims, activations, **hyperparameters, last_activation="linear")

    # validate on validation set
    A_val, _ = custom_model_forward(X_val, model_parameters, activations, "linear")
    val_cost = BCE_WithLogitsLoss(A_val, Y_val, from_logits=True)
    print(f"Validation cost: {val_cost}")
    print(f"Training completed with total time of {time.time() - start_time} seconds!")
    
    # TODO: save model_parameters to file
    
