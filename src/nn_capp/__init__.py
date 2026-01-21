"""
Classic Neural Network Implementation using classic_nn package
"""
import time
import sys
import os
from glob import glob

from src.load.read_produce_dataset import load_yaml_classes,process_dataset_batches, scale_data
from src.classic_nn import update_parameters, initialize_parameters_deep, custom_model_forward, custom_model_backward, BCE_WithLogitsLoss

import numpy as np
import gc



def forward_and_backward_propagation(X, Y, parameters, activations, learning_rate=0.0075, num_classes=1, last_activation="sigmoid"):
    """
    Running forward and backward propagation on a single batch of data
    Returns cost, grads, parameters
    """    
    print("Running forward and backward propagation for the number of classes: {}".format(num_classes))

    A, caches = custom_model_forward(X, parameters, activations, num_classes, apply_sigmoid=(last_activation == "sigmoid"))
    cost = BCE_WithLogitsLoss(A, Y, from_logits=(last_activation == "linear"))
    grads = custom_model_backward(A, Y, caches, activations, last_activation)
    parameters = update_parameters(parameters, grads, learning_rate)
    
    return cost, grads, parameters
    

# NOTE: Do not this model for mini batch training.
def custom_layer_model(X, Y, layers_dims, activations, learning_rate=0.0075, num_classes=1, num_iterations=3000, print_cost=False, parameters=None, last_activation="sigmoid"):
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
    
    parameters = parameters if parameters else initialize_parameters_deep(layers_dims)
    
    for i in range(num_iterations):
        A, caches = custom_model_forward(X, parameters, activations, num_classes, apply_sigmoid=(last_activation == "sigmoid"))
        cost = BCE_WithLogitsLoss(A, Y, from_logits=(last_activation == "linear"))
        costs.append(cost)
        grads = custom_model_backward(A, Y, caches, activations, last_activation)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    
    return parameters


def train_model_mini_batch(print_cost=False):
    """
    Train the model using mini-batch gradient descent. NOTE: input data has been split into train, dev in advance. See process_dataset function.
    Therefore, all X, Y are already scaled and split into train, dev prior to training.
    """
    batch_size = 1024        
    hyperparameters = {
        "epochs": 1000,
        "learning_rate": 0.0075,
        "num_iterations": 3000,
        "batch_size": 1024,        
        "target_size": 64,
        "last_activation": "linear",
        "num_classes": 1
    }
      

    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))    
    print("Loading dataset in batches with size {}...".format(batch_size))
    
    start_time = time.time()

    project_root = os.getcwd()
    images_root = os.path.join(project_root, "assets/produce_dataset/LVIS_Fruits_And_Vegetables")
    yaml_path = os.path.join(project_root, "assets/produce_dataset/LVIS_Fruits_And_Vegetables/data.yaml")

    data = load_yaml_classes(yaml_path)
    train_images = os.path.join(images_root, data["train"])    
    image_paths = sorted(glob(os.path.join(train_images, "*.jpg")) +
                         glob(os.path.join(train_images, "*.png")))

    labels_dir = os.path.join(images_root, data["train"]).replace("images", "labels")

    input_size = hyperparameters["target_size"] * hyperparameters["target_size"] * 3
    output_size = len(data["names"])
    layers_dims = [input_size, 2048, 1024, 512, 256, 128, output_size]
    activations = ["relu"] * (len(layers_dims) - 1) + ["linear"]
    hyperparameters["num_classes"] = output_size
    
    parameters = initialize_parameters_deep(layers_dims)
    
    for epoch in range(hyperparameters["epochs"]):
        epoch_cost = 0
        num_batches = (len(image_paths) + batch_size - 1) // batch_size

        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1} "
                    f"(images {batch_start}-{batch_end-1})")
            X_batch, Y_batch = process_dataset_batches(image_paths, labels_dir, starting_image_path_index = batch_start, target_size=hyperparameters["target_size"], batch_size = batch_size)
            X_batch = scale_data(X_batch, method='minmax')  # Scale to [0, 1]    
            cost, _, _ = forward_and_backward_propagation(X_batch, Y_batch, parameters, activations, num_classes=hyperparameters["num_classes"], learning_rate=hyperparameters["learning_rate"], last_activation=hyperparameters["last_activation"])                       

            # clean up memory
            del X_batch, Y_batch
            gc.collect()

            epoch_cost += cost / num_batches
        # print cost after every 10 epochs
        if print_cost and (epoch % 10 == 0 or epoch == hyperparameters["epochs"] - 1):
            print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Cost: {epoch_cost:.4f}")
        
    print(f"Training completed with total time of {time.time() - start_time} seconds!")
    # TODO: save model_parameters to file
    
   
