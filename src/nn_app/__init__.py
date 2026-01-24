"""
Classic Neural Network Implementation using classic_nn package
"""
import time
import sys
import os
from glob import glob

from src.load.read_produce_dataset import load_yaml_classes,batch_generator
from src.classic_nn import update_parameters, initialize_parameters_deep, custom_model_forward, custom_model_backward, BCE_WithLogitsLoss, forward_and_backward_propagation

import gc


# NOTE: Do not this model for mini batch training.
def multi_label_model(X, Y, layers_dims, activations, learning_rate=0.0075, num_classes=1, num_iterations=3000, print_cost=False, parameters=None, last_activation="sigmoid"):
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

# Train YoLo image classification model. NOTE: this model is not used for training. It is just for testing
def train_mlayer_mini_batch(print_cost=False):
    """
    Train the model using mini-batch gradient descent. NOTE: input data has been split into train, dev in advance. See process_dataset function.
    Therefore, all X, Y are already scaled and split into train, dev prior to training.
    """
    
    hyperparameters = {
        "epochs": 1, # TODO: change to 1000
        "learning_rate": 0.0075,
        "num_iterations": 3000,
        "batch_size": 1024,        
        "target_size": 64,
        "last_activation": "linear",
        "num_classes": 1
    }
      

    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))    
    print("Loading dataset in batches with size {}...".format(hyperparameters["batch_size"]))
    
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
    total_images = len(image_paths)
    estimated_batches = (total_images // hyperparameters["batch_size"]) + (hyperparameters["batch_size"] -1)
    for epoch in range(hyperparameters["epochs"]):
        epoch_cost = 0
        num_batches = 0       
        for X_batch, Y_batch in batch_generator(
                image_paths, labels_dir, 
                hyperparameters["batch_size"], 
                hyperparameters["target_size"]
            ):
            cost, _, _ = forward_and_backward_propagation(X_batch, Y_batch, parameters, activations,
             num_classes=hyperparameters["num_classes"],
             learning_rate=hyperparameters["learning_rate"], last_activation=hyperparameters["last_activation"])               
                        
            # clean up memory
            del X_batch, Y_batch         
            num_batches += 1
            epoch_cost += cost

            # Print progress
            print(f"\rEpoch {epoch+1}: {num_batches/estimated_batches*100:.1f}% complete", end='', flush=True)

        epoch_cost  = epoch_cost / num_batches
        
        gc.collect()        
        if print_cost:
            print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Cost: {epoch_cost:.4f}")
        
        # add progress percentage
        progress = (epoch + 1) / hyperparameters["epochs"]
        print(f"Progress: {progress:.2%}")


    print(f"Training completed with total time of {time.time() - start_time} seconds!")
    # TODO: save model_parameters to file
    

