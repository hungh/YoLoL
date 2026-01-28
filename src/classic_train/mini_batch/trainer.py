import time
import gc

from matplotlib import pyplot as plt

from src.classic_nn.batch_norm import initialize_parameters_deep, gradient_descent
from src.classic_nn.batch_norm.forward import custom_model_forward
from src.classic_nn.math import sigmoid
from src.classic_nn.utils import pca_plot_dataset
from src.classic_nn.optimizers import OptimizerFactory
from src.classic_nn.utils import plot_costs, plot_decision_boundary
from src.classic_nn.utils.eval import predict, evaluate_model
from src.classic_nn.optimizers.adam import Adam

from .metrics import multi_label_metrics
from .utils import mini_batch_generator as mb_gen   



class MiniBatchTrainer:
    def __init__(self, X_train, Y_train, layers_dims, learning_rate=0.001, activations=None, num_classes=1,
     mini_batch_size=64, num_epochs=1000, print_cost=True, plot_input_data=False):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = None
        self.Y_val = None
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.activations = activations
        self.num_classes = num_classes
        self.mini_batch_size = mini_batch_size
        self.num_epochs = num_epochs
        self.print_cost = print_cost
        self.parameters = None
        self.costs = []
        self.optimizer_name = None     
        self.lambda_reg = 0.01   
        self.logits = False
        self.prediction_threshold = 0.5
        self.plot_input_data = plot_input_data

    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = kwargs
    
    def set_validation_data(self, X_val, Y_val):
        self.X_val = X_val
        self.Y_val = Y_val
    
    def set_prediction_threshold(self, threshold):
        self.prediction_threshold = threshold

    def set_regularization(self, lambda_reg):
        """
        Set the regularization parameter.
        
        Args:
            lambda_reg: Regularization parameter (default: 0.01)
        """
        self.lambda_reg = lambda_reg

    def enable_logits(self):
        self.logits = True        
    
    def adjust_predictions(self, A, last_activation):
        """
        Adjust predictions after applying sigmoid.
        
        Args:
            A: Activations from the last layer
            last_activation: Activation function of the last layer
            
        Returns:
            Y_pred: Adjusted predictions
        """
        # adjust predictions after apply sigmoid for a case when last activation is linear and logits are used
        if self.logits:
            print(f"[INFO] Using logits: {self.logits}, last activation: {last_activation}")
            Y_pred, _ = sigmoid(A)
        else:
            Y_pred = A
        return Y_pred

    def validate_hyperparameters(self):
        assert self.mini_batch_size > 0, "Mini batch size must be greater than 0"
        assert self.num_epochs > 0, "Number of epochs must be greater than 0"
        assert self.learning_rate > 0, "Learning rate must be greater than 0"
        assert self.num_classes > 0, "Number of classes must be greater than 0"
        assert self.layers_dims is not None, "Layer dimensions must be provided"
        assert len(self.layers_dims) > 0, "Layer dimensions must not be empty"
        assert self.activations is not None, "Activations must be provided"
        assert len(self.activations) == (len(self.layers_dims) - 1), "Activations must match layer dimensions"
        # make sure trainig and validation datasets have been set
        assert self.X_train is not None, "Training dataset must be provided"
        assert self.Y_train is not None, "Training labels must be provided"        
        # make sure trainig and validation datasets have the same number of features
        if self.X_val is not None and self.Y_val is not None:
            assert self.X_train.shape[0] == self.X_val.shape[0], "Training and validation datasets must have the same number of features"

    def evaluate_model_wrapper(self, parameters, last_activation):
        # check if this is a multi label model
        if self.num_classes > 1:
            A_train, _ = custom_model_forward(self.X_train, parameters, self.activations, self.num_classes, apply_sigmoid=False)
            metrics_train = multi_label_metrics(A_train, self.Y_train, prediction_threshold=self.prediction_threshold)
            
            A_val, _ = custom_model_forward(self.X_val, parameters, self.activations, self.num_classes, apply_sigmoid=False)
            metrics_val = multi_label_metrics(A_val, self.Y_val, prediction_threshold=self.prediction_threshold)
            
            print(f"Training metrics:")
            print(f"last activation: {last_activation}")
            print(f"Training metrics: {metrics_train}")
            print(f"Validation metrics: {metrics_val}")
        else:
            print(f"Training metrics:")
            print(f"last activation: {last_activation}")
            evaluate_model(self.X_train, self.Y_train, parameters, self.activations, num_classes=self.num_classes, last_activation=last_activation, 
            prediction_threshold=self.prediction_threshold, adjust_prediction_func=self.adjust_predictions)

            print(f"\n\nDev metrics:")
            evaluate_model(self.X_val, self.Y_val, parameters, self.activations, num_classes=self.num_classes, last_activation=last_activation, 
            prediction_threshold=self.prediction_threshold, adjust_prediction_func=self.adjust_predictions)

    # to be implemented by subclasses, additional updates to parameters (such as ones used in Adam optimizer)
    def abstract_update_mb_t(self, parameters, mb_t):
        pass

    def __str__(self):
        return f"MiniBatchTrainer(optimizer={self.optimizer_name}, learning_rate={self.learning_rate}, mini_batch_size={self.mini_batch_size}, num_epochs={self.num_epochs})"
        
    def train(self):
        """
        Train the model using mini-batch gradient descent.
        """
        # print input shape
        print(f"Input shape: {self.X_train.shape}")
        print(f"Output shape: {self.Y_train.shape}")

        # validate minimum hyperparameters
        self.validate_hyperparameters()
    

        # plot dataset
        if self.plot_input_data:
            pca_plot_dataset(self.X_train, self.Y_train)        

        # print all hyperparameters
        print("Hyperparameters:")
        print(self)
        print("-" * 50); print("\n")
        

        last_activation = self.activations[-1]

        start_time = time.time()
        
        # initialize optimizer
        print(f"Optimizer arguments: {self.optimizer_kwargs}")
        optimizer_instance = OptimizerFactory(self.optimizer_name).get_optimizer(**self.optimizer_kwargs) if self.optimizer_kwargs else OptimizerFactory(self.optimizer_name).get_optimizer()
        parameters = initialize_parameters_deep(self.layers_dims, optimizer_instance)  
        
        estimated_batches = (self.X_train.shape[1] + self.mini_batch_size - 1) // self.mini_batch_size
        print(f"Estimated batches per epoch: {estimated_batches}")

        print_cost_every = max(1, self.num_epochs // 100)

        for epoch in range(self.num_epochs):
            epoch_cost = 0
            num_batches = 0    
            mb_t = 1 # the mini batch iteration index if Adam is used
            for X_batch, Y_batch in mb_gen(self.X_train, self.Y_train, self.mini_batch_size):            

                self.abstract_update_mb_t(parameters, mb_t)
                
                # the parameters will be updated in this function
                cost, _, _ = gradient_descent(X_batch, Y_batch, parameters, self.activations,
                num_classes=self.num_classes,
                learning_rate=self.learning_rate, last_activation=last_activation, lambda_reg=self.lambda_reg)               
                            
                # clean up memory
                del X_batch, Y_batch         
                num_batches += 1
                epoch_cost += cost
                mb_t += 1

                # Print progress
                print(f"\rEpoch {epoch+1}: {num_batches/estimated_batches*100:.1f}% complete", end='', flush=True)

            epoch_cost  = epoch_cost / num_batches
            
            # clean up memory after print_cost_every epochs
            if epoch % print_cost_every == 0:
                gc.collect()        
                self.costs.append(epoch_cost)
                if self.print_cost:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Cost: {epoch_cost:.4f}")       
            
            # add progress percentage
            if epoch % (self.num_epochs // print_cost_every) == 0:
                progress = (epoch + 1) / self.num_epochs
                print(f"Progress: {progress*100:.1f}%")

        print(f"Training completed with total time of {time.time() - start_time} seconds!")
       
        self.evaluate_model_wrapper(parameters, last_activation)
        
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        # plot costs
        plot_costs(self.costs, title="Training Costs", ax=ax1)
        
        # plot decision boundary
        if self.num_classes < 2:
            plot_decision_boundary(self.X_val, self.Y_val, predict, parameters, self.activations, self.num_classes, ax2)
        else:
            print("Skipping decision boundary plot for multi-class classification")
        plt.show()
            
        return parameters






