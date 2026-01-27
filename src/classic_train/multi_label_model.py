"""
Batch Normalization Optimized Mini Batch Model
"""

from src.classic_nn.data_utils import generate_multilabel_dataset
from .mini_batch.trainer import MiniBatchTrainer

def train_model():
    """
    Train a multi-label classification model using mini-batch gradient descent.
    
    This function sets up and trains a neural network for multi-label classification
    where each sample can belong to multiple classes simultaneously.
    """
    # Generate multi-label dataset
    n_classes = 3
    X_train, Y_train, X_dev, Y_dev = generate_multilabel_dataset(n_samples=1000, n_features=10, n_classes=n_classes, avg_labels=(n_classes - 1))

    print(f"X_train shape: {X_train.shape}")
    print(f"X_dev shape: {X_dev.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_dev shape: {Y_dev.shape}")

    # hyperparameters
    learning_rate = 0.003
    num_epochs = 3000
    layers_dims = [X_train.shape[0], 64, 32, 16, 8, n_classes]
    activations = ["relu"] * (len(layers_dims) - 2) + ["linear"]
    num_classes = n_classes
    mini_batch_size = X_train.shape[1] // 2 # 64
    lambda_reg = 0.01
    print_cost = True

    mini_batch_trainer = MiniBatchTrainer(
        X_train=X_train,
        Y_train=Y_train,
        layers_dims=layers_dims,
        learning_rate=learning_rate,
        activations=activations,
        num_classes=num_classes,
        mini_batch_size=mini_batch_size,
        num_epochs=num_epochs,
        print_cost=print_cost
    )
    # last layer is linear for multi-label classification
    mini_batch_trainer.enable_logits()
    mini_batch_trainer.set_prediction_threshold(0.5)
    mini_batch_trainer.set_validation_data(X_dev, Y_dev)
    mini_batch_trainer.set_optimizer("momentum", beta=0.9)    
    mini_batch_trainer.set_regularization(lambda_reg)
    mini_batch_trainer.train()

