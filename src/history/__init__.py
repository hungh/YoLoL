"""
History module for tracking training progress and results.
"""
import numpy as np
from pathlib import Path


# Optional imports with error handling
try:
    from torch.utils.tensorboard import SummaryWriter        
    TENSORBOARD_AVAILABLE = True
except ImportError as e:
    TENSORBOARD_AVAILABLE = False
    print(f"Warning: TensorBoard dependencies not available: {e}")

class TrainingHistoryWriter:
    """Class to track training history and results."""
    WRITERS = ["cost", "parameters", "gradients", "metrics"]

    def __init__(self, log_dir=None, env_config=None):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard dependencies required. Install with: pip install torch tbparse")

        self.env_config = env_config
        self.base_dir = Path(log_dir) if log_dir else env_config.get_log_dir()
        self.is_history_enabled = self.env_config.get_history_enabled()

        if self.is_history_enabled == True:
            self.cost_writer = SummaryWriter(str(self.base_dir / "cost"))
            self.parameters_writer = SummaryWriter(str(self.base_dir / "parameters"))
            self.gradients_writer = SummaryWriter(str(self.base_dir / "gradients"))
            self.metrics_writer = SummaryWriter(str(self.base_dir / "metrics"))        
        else:
            self.cost_writer = None
            self.parameters_writer = None
            self.gradients_writer = None
            self.metrics_writer = None
            print("History is disabled")
        

    def is_enabled(self):
        return self.is_history_enabled
        
    def _record_history(self, writer_name, record_object, epoch, mini_batch=None, is_grad_norm=False):
        if self.is_history_enabled == False:
            return
        
        if writer_name not in TrainingHistoryWriter.WRITERS:
            raise ValueError(f"Invalid writer name: {writer_name}")
        
        mini_batch_str = f'/batch_no_{mini_batch}' if mini_batch is not None else ''
        writer = getattr(self, f"{writer_name}_writer")

        if writer is None:
            return
        
        if not isinstance(record_object, dict):
            writer.add_scalar(f"{mini_batch_str}/{writer_name}", record_object, epoch)
            return
        
        for key, value in record_object.items():
            if np.isscalar(value):
                writer.add_scalar(f"{mini_batch_str}/{writer_name}/{key}", value, epoch)
            else:
                writer.add_scalar(f"{mini_batch_str}/{writer_name}/{key}_mean", np.mean(value), epoch)
                writer.add_scalar(f"{mini_batch_str}/{writer_name}/{key}_std", np.std(value), epoch)
                if is_grad_norm:
                    writer.add_scalar(f"{mini_batch_str}/{writer_name}/{key}_norm", np.linalg.norm(value), epoch)


    def add_parameters(self, parameters, epoch, mini_batch=None):
        self._record_history("parameters", parameters, epoch, mini_batch)       

    
    def add_gradients(self, gradients, epoch, mini_batch=None):
        self._record_history("gradients", gradients, epoch, mini_batch, is_grad_norm = True)      
    
    def add_cost(self, cost, epoch, mini_batch=None):
        self._record_history("cost", cost, epoch, mini_batch)
    
    def add_metrics(self, metrics, epoch, mini_batch=None):
        self._record_history("metrics", metrics, epoch, mini_batch)
    
    def close(self):
        """Close the writer"""
        if self.is_history_enabled == True:
            self.cost_writer.close()
            self.parameters_writer.close()
            self.gradients_writer.close()
            self.metrics_writer.close()



