import time
from src.classic_nn.batch_norm import initialize_parameters_deep, forward_and_backward_propagation, sigmoid, custom_model_forward
import gc
from src.classic_nn.utils import plot_dataset, mini_batch_generator, multi_label_metrics
from src.classic_nn.optimizers import OptimizerFactory
from ..classic_nn.utils import plot_costs

class MiniBatchTrainer:
    def __init__(self, X_train, Y_train, layers_dims, learning_rate=0.001, activations=None, num_classes=1, mini_batch_size=64, num_epochs=1000, print_cost=True):
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

    def set_optimizer(self, optimizer_name, **kwargs):
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = kwargs
    
    def set_validation_data(self, X_val, Y_val):
        self.X_val = X_val
        self.Y_val = Y_val

    def set_regularization(self, lambda_reg):
        """
        Set the regularization parameter.
        
        Args:
            lambda_reg: Regularization parameter (default: 0.01)
        """
        self.lambda_reg = lambda_reg

    def logits(self):
        self.logits = True
        return self
    
    def adjust_predictions(self, A, last_activation):
        """
        Adjust predictions after applying sigmoid.
        
        Args:
            A: Activations from the last layer
            last_activation: Activation function of the last layer
            
        Returns:
            Y_pred: Adjusted predictions
        """
        # adjust predictions after apply sigmoid
        if self.logits and last_activation == "sigmoid":
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
        assert len(self.activations) == len(self.layers_dims), "Activations must match layer dimensions"
        # make sure trainig and validation datasets have been set
        assert self.X_train is not None, "Training dataset must be provided"
        assert self.Y_train is not None, "Training labels must be provided"        
        # make sure trainig and validation datasets have the same number of features
        if self.X_val is not None and self.Y_val is not None:
            assert self.X_train.shape[0] == self.X_val.shape[0], "Training and validation datasets must have the same number of features"

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
        if self.print_cost:
            plot_dataset(self.X_train, self.Y_train)        

        # print all hyperparameters
        print("Hyperparameters:")
        print(self)
        print("-" * 50); print("\n")
        

        last_activation = self.activations[-1]

        start_time = time.time()
        
        optimizer_instance = OptimizerFactory(self.optimizer_name).get_optimizer(**self.optimizer_kwargs)
        parameters = initialize_parameters_deep(self.layers_dims, optimizer_instance)  
        
        estimated_batches = (self.X_train.shape[1] + self.mini_batch_size - 1) // self.mini_batch_size

        for epoch in range(self.num_epochs):
            epoch_cost = 0
            num_batches = 0       
            for X_batch, Y_batch in mini_batch_generator(self.X_train, self.Y_train, self.mini_batch_size):
                # the parameters will be updated in this function
                cost, _, _ = forward_and_backward_propagation(X_batch, Y_batch, parameters, self.activations,
                num_classes=self.num_classes,
                learning_rate=self.learning_rate, last_activation=last_activation, lambda_reg=self.lambda_reg)               
                            
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
                self.costs.append(epoch_cost)
                if self.print_cost:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Cost: {epoch_cost:.4f}")       
            
            # add progress percentage
            if epoch % (self.num_epochs // 10) == 0:
                progress = (epoch + 1) / self.num_epochs
                print(f"Progress: {progress*100:.1f}%")
            


        print(f"Training completed with total time of {time.time() - start_time} seconds!")

        # plot costs
        plot_costs(self.costs, title="Training Costs")

        # calculate the accuracy of train set
        A_train, _ = custom_model_forward(self.X_train, parameters, self.activations, self.num_classes, apply_sigmoid=(last_activation == "sigmoid"))
        Y_pred_train = self.adjust_predictions(A_train, last_activation)
        metrics_train = multi_label_metrics(Y_pred_train, self.Y_train)
        print("Train Metrics:")
        for k, v in metrics_train.items():
            print(f"{k}: {v:.4f}")
        
        # inferenace the model on validation set
        if self.X_val is not None and self.Y_val is not None:
            A, _ = custom_model_forward(self.X_val, parameters, self.activations, self.num_classes, apply_sigmoid=(last_activation == "sigmoid"))
            # adjust predictions based on last activation
            Y_pred = self.adjust_predictions(A, last_activation)
            
            metrics = multi_label_metrics(Y_pred, self.Y_val)
            print("Validation Metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")           
        else:
            print("No validation data provided, skipping validation metrics")
        return parameters


