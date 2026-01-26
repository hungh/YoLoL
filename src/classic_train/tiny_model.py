from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
import numpy as np
import time
from src.classic_nn import initialize_parameters_deep, forward_and_backward_propagation, sigmoid, custom_model_forward
import matplotlib.pyplot as plt
import gc

from src.classic_nn.utils import plot_dataset, multi_label_metrics


def load_tiny_dataset(n_samples=1000, n_features=32*32*3, n_classes=3, avg_labels=3, test_size=0.2, random_state=42):
    """
    Load a tiny multi-label dataset for testing.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features per sample
    n_classes : int
        Number of classes
    avg_labels : int
        Average number of labels per sample
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, Y_train, X_test, Y_test : tuple
        Training and test datasets
    """    
    # avg_labels    # Average number of objects per image

    # Generate the data
    X, Y = make_multilabel_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_labels=avg_labels,  # This is the "Poisson" mean for label count
        allow_unlabeled=False, # Ensures every 'image' has at least one object
        random_state=random_state
    )

    print(f"Data set has been initialized with {n_samples} samples, {n_features} features, {n_classes} classes, {avg_labels} average labels per sample")
    # split the dataset into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, shuffle=True)
    
    print("Train label distribution:", np.mean(Y_train, axis=0))
    print("Test label distribution:", np.mean(Y_test, axis=0))

    # Returns X as (Features, Samples) and Y as (Classes, Samples)    
    return X_train.T, Y_train.T, X_test.T, Y_test.T 


def tin_batch_generator(X_train, Y_train, batch_size):
    """
    Generate batches of data for training.
    """
    for i in range(0, X_train.shape[1], batch_size):
        yield X_train[:, i:i+batch_size], Y_train[:, i:i+batch_size]

"""
NOTE: the model overfitting on train set. See exec metrics below
Train Metrics:
exact_match: 1.0000
hamming_loss: 0.0000
f1_score: 1.0000
avg_predicted: 0.5643
avg_actual: 0.5643
Test Metrics:
exact_match: 0.1540
hamming_loss: 0.3182
f1_score: 0.7600
avg_predicted: 0.7593
avg_actual: 0.5664

"""
def train_model_mini_batch(X_train, Y_train, X_test, Y_test, print_cost=False, num_classes=1, enable_plot=False):
    """
    Train the tiny model using mini-batch gradient descent.
    """
    assert X_train.shape[0] == X_test.shape[0], "Input feature dimension mismatch"
    assert Y_train.shape[0] == Y_test.shape[0], "Output dimension mismatch"

    # check for data leakage between train and test
    # Ensure no data leakage between train/test
    print("Are there any overlapping samples?", len(set(X_train.T.tobytes()) & set(X_test.T.tobytes())) > 0)    
    
    # print input shape
    print(f"Input shape: {X_train.shape}")
    print(f"Output shape: {Y_train.shape}")
  

    # plot dataset
    if enable_plot:
        plot_dataset(X_train, Y_train)
        # plot_dataset(X_test, Y_test)
        
    
    hyperparameters = {
        "epochs": 500, 
        "learning_rate": 0.0075, # TODO: will use decay learning rate        
        "batch_size": 64,                
        "last_activation": "linear",
        "num_classes": num_classes,
        "lambda_reg": 0.01 
    }    

    # print all hyperparameters
    print("Hyperparameters:")
    for k, v in hyperparameters.items():
        print(f"{k}: {v}")
    print("-" * 50); print("\n")
    
    start_time = time.time()

    input_size = X_train.shape[0]
    # layers_dims = [input_size, 256, 128, 64, 32, num_classes]
    layers_dims = [input_size, 64, 32, 16, num_classes]
    # TODO: fix this issue with an assumption that the input is counted as one layer
    activations = ["relu"] * (len(layers_dims) - 1) + ["linear"]
        
    parameters = initialize_parameters_deep(layers_dims)  
    
    estimated_batches = (X_train.shape[1] + hyperparameters["batch_size"] - 1) // hyperparameters["batch_size"]

    for epoch in range(hyperparameters["epochs"]):
        epoch_cost = 0
        num_batches = 0       
        for X_batch, Y_batch in tin_batch_generator(X_train, Y_train, hyperparameters["batch_size"]):
            # the parameters will be updated in this function
            cost, _, _ = forward_and_backward_propagation(X_batch, Y_batch, parameters, activations,
             num_classes=hyperparameters["num_classes"],
             learning_rate=hyperparameters["learning_rate"], last_activation=hyperparameters["last_activation"], lambda_reg=hyperparameters["lambda_reg"])               
                        
            # clean up memory
            del X_batch, Y_batch         
            num_batches += 1
            epoch_cost += cost

            # Print progress
            print(f"\rEpoch {epoch+1}: {num_batches/estimated_batches*100:.1f}% complete", end='', flush=True)

        epoch_cost  = epoch_cost / num_batches
        
        # clean up memory after 100 epochs
        if epoch % 10 == 0:
            gc.collect()        
            if print_cost:
                print(f"Epoch {epoch+1}/{hyperparameters['epochs']}, Cost: {epoch_cost:.4f}")       
        
        # add progress percentage
        if epoch % (hyperparameters["epochs"] // 10) == 0:
            progress = (epoch + 1) / hyperparameters["epochs"]
            print(f"Progress: {progress*100:.1f}%")
        


    print(f"Training completed with total time of {time.time() - start_time} seconds!")

    # calculate the accuracy of train set
    A_train, _ = custom_model_forward(X_train, parameters, activations, num_classes, apply_sigmoid=(hyperparameters["last_activation"] == "sigmoid"))
    metrics_train = multi_label_metrics(A_train, Y_train)
    print("Train Metrics:")
    for k, v in metrics_train.items():
        print(f"{k}: {v:.4f}")
    
    # inferenace the model on test set
    A, _ = custom_model_forward(X_test, parameters, activations, num_classes, apply_sigmoid=(hyperparameters["last_activation"] == "sigmoid"))
    
    # calculate accuracy after apply sigmoid
    # print(f"A shape: {A.shape} and its 5 first values: {A[:1]}")
    Y_pred, _ = sigmoid(A)
    metrics = multi_label_metrics(Y_pred, Y_test)
    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def train_model():
    number_cls = 5
    plot_yes = False
    print(f"Plotting: {plot_yes}")
    X_train, Y_train, X_test, Y_test = load_tiny_dataset(n_samples=15000, n_classes=number_cls)
    train_model_mini_batch(X_train, Y_train, X_test, Y_test, print_cost=True, num_classes=number_cls, enable_plot=plot_yes)


