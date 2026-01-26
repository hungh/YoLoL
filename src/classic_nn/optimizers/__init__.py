class Optimizers:
    """
    Optimizers class for selecting and initializing optimizers.
    """
    
    def __init__(self, name: str):
       self.name = name
       self.current_optimizer = None
       

    def get_optimizer(self, **kwargs):
        if self.name == "adam":
            from .adam import Adam
            self.current_optimizer = Adam(**kwargs)            
        elif self.name == "momentum":
            from .momentum import Momentum
            self.current_optimizer = Momentum(**kwargs)                        
        else:
            raise ValueError("Invalid optimizer name")

        return self.current_optimizer

    def validate_instance(self):
        if self.current_optimizer is None:
            raise ValueError("Optimizer not initialized. Call get_optimizer() first.")
       

    def initialize_parameters(self, parameters, layer_dims):
        self.validate_instance(self)
        self.current_optimizer.initialize_parameters(parameters, layer_dims)

    
    def __str__(self):
        return self.name
