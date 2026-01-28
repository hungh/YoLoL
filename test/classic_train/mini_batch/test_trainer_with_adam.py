# test_trainer_with_adam.py
import numpy as np
from src.classic_train.mini_batch.trainer_with_adam import TrainerWithAdam
from src.classic_nn.optimizers.adam import Adam

def test_trainer_creation():
    """Test that TrainerWithAdam can be created"""
    X_train = np.random.rand(2, 10)
    Y_train = np.random.randint(0, 2, (1, 10))
    
    trainer = TrainerWithAdam(
        X_train=X_train,
        Y_train=Y_train,
        layers_dims=[2, 4, 1],
        learning_rate=0.01,
        activations=["relu", "sigmoid"],
        num_classes=1,
        mini_batch_size=5,
        num_epochs=1,
        print_cost=False
    )
    
    assert trainer.optimizer_name == "adam"

def test_method_overriding():
    """Test that abstract_update_mb_t method is properly overridden"""
    X_train = np.random.rand(2, 10)
    Y_train = np.random.randint(0, 2, (1, 10))
    
    trainer = TrainerWithAdam(
        X_train=X_train,
        Y_train=Y_train,
        layers_dims=[2, 4, 1],
        learning_rate=0.01,
        activations=["relu", "sigmoid"],
        num_classes=1,
        mini_batch_size=5,
        num_epochs=1,
        print_cost=False
    )
    
    # Test method behavior
    test_parameters = {}
    test_mb_t = 5
    
    trainer.abstract_update_mb_t(test_parameters, test_mb_t)
    
    assert Adam.MB_T in test_parameters
    assert test_parameters[Adam.MB_T] == test_mb_t