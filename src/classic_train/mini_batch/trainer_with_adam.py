# a subclass of MiniBatchTrainer that uses Adam optimizer
from .trainer import MiniBatchTrainer
from src.classic_nn.optimizers.adam import Adam

class TrainerWithAdam(MiniBatchTrainer):
    def __init__(self, X_train, Y_train, layers_dims, learning_rate=0.001, activations=None, num_classes=1,
     mini_batch_size=64, num_epochs=1000, print_cost=True, plot_input_data=False):
        super().__init__(X_train, Y_train, layers_dims, learning_rate, activations, num_classes, mini_batch_size, num_epochs, print_cost, plot_input_data)
        self.optimizer_name = "adam"      
        self.optimizer_kwargs = None
        
        
    def update_adam_hyperparameters(self, beta1, beta2, epsilon):
        """
        Update Adam hyperparameters. This function is used for rare fine tuning of Adam hyperparameters
        
        Args:
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    
    def abstract_update_mb_t(self, parameters, mb_t):
        """
        Update the iteration number of the mini-batch. This function is used for rare fine tuning of Adam hyperparameters
        """
        parameters[Adam.MB_T] = mb_t