class OptimizerFactory:
    """
    Optimizers class for selecting and initializing optimizers.
    """
    
    def __init__(self, name: str):
       self.name = name
       self.current_optimizer = None
       

    def get_optimizer(self, **kwargs):
        """
        Get the optimizer instance.
        Input:
            **kwargs: Additional arguments to pass to the optimizer.
        Returns:
            The optimizer instance.
        """
        if self.name == "adam":
            from .adam import Adam
            self.current_optimizer = Adam(**kwargs)            
        elif self.name == "momentum":
            from .momentum import Momentum
            self.current_optimizer = Momentum(**kwargs)       
        elif self.name == "rms_prop":
            from .rms_prop import RMSProp
            self.current_optimizer = RMSProp(**kwargs)         
        else:
            raise ValueError(f"Invalid optimizer name: {self.name}")

        return self.current_optimizer

    def validate_instance(self):
        if self.current_optimizer is None:
            raise ValueError("Optimizer not initialized. Call get_optimizer() first.")
       

    def initialize_parameters(self, parameters, layer_dims):
        """
        Initialize the optimizer parameters.
        Input:
            parameters: The parameters to initialize.
            layer_dims: The dimensions of the layers.
        """
        self.validate_instance()
        self.current_optimizer.initialize_parameters(parameters, layer_dims)

    
    def __str__(self):
        """
        Return the name of the optimizer.
        Returns:
            The name of the optimizer.
        """
        return self.name
