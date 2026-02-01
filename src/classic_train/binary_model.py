"""
Batch Normalization Optimized Mini Batch Model
"""
from .mini_batch.trainer import MiniBatchTrainer
from ..classic_nn.data_utils import generate_binary_classification_data
from src.learning_rate import LearningRateDecay
from ..history import TrainingHistoryWriter
from dependency_injector.wiring import Provide, inject
from ..di.containers import Container    

@inject
def train_model(history_writer: TrainingHistoryWriter = Provide[Container.history_writer]):
    """
    Train a binary classification model using mini-batch gradient descent.
    
    This function sets up and trains a neural network for binary classification
    with a single output neuron.
    """    
    
    # Generate binary classification dataset
    X, Y = generate_binary_classification_data()

    # split the data set into train and dev
    from sklearn.model_selection import train_test_split
    X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

    # normalize the data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_dev = sc.transform(X_dev)

    # reshape X
    X_train = X_train.T
    X_dev = X_dev.T

    # reshape Y
    Y_train = Y_train.reshape(1, -1)
    Y_dev = Y_dev.reshape(1, -1)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_dev shape: {X_dev.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_dev shape: {Y_dev.shape}")

    # hyperparameters
    learning_rate = 0.01
    num_epochs = 1000
    layers_dims = [X_train.shape[0], 8, 4, 1]
    activations = ["relu", "relu", "sigmoid"]
    num_classes = 1
    mini_batch_size = X_train.shape[1] #64
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
        print_cost=print_cost,
        plot_input_data=True,
        history_writer=history_writer
    )

    mini_batch_trainer.set_validation_data(X_dev, Y_dev)
    mini_batch_trainer.set_optimizer("momentum", beta=0.9)
    mini_batch_trainer.set_regularization(lambda_reg)
    mini_batch_trainer.set_learning_rate_decay(LearningRateDecay.get_decay_function("std"), initial_lr=learning_rate, decay_rate=1.0)
    # mini_batch_trainer.set_learning_rate_decay(LearningRateDecay.get_decay_function("step"))
    mini_batch_trainer.train()
