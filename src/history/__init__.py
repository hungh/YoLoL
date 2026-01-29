"""
History module for tracking training progress and results.
"""
import numpy as np
import yaml
from pathlib import Path


# Optional imports with error handling
try:
    from torch.utils.tensorboard import SummaryWriter
    from tbparse import SummaryReader
    TENSORBOARD_AVAILABLE = True
except ImportError as e:
    TENSORBOARD_AVAILABLE = False
    print(f"Warning: TensorBoard dependencies not available: {e}")

class TrainingHistoryWriter:
    """Class to track training history and results."""
    
    def __init__(self, log_dir=None):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard dependencies required. Install with: pip install torch tbparse")
        
        self.log_dir = Path(log_dir) if log_dir else self.get_proj_home_dir() / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.is_history_enabled = self.is_enabled()

    def is_enabled(self):
        is_enabled = True
        prj_home_dir = self.get_proj_home_dir()
        config_path = prj_home_dir / "environments.yaml"        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                is_enabled = config.get('default', {}).get('HISTORY_ENABLED', True)
        else:
            print("environments.yaml not found, history is enabled by default")

        print(f"History enabled: {is_enabled}")
        return is_enabled
        

    def get_proj_home_dir(self):                      
        return Path.cwd()
        
    def add_parameters(self, parameters, epoch, mini_batch=None):
        if not self.is_history_enabled:
            return
        """Track parameter values"""
        mini_batch_str = f'/batch_no_{mini_batch}' if mini_batch is not None else ''
        for key, value in parameters.items():
            if np.isscalar(value):
                self.writer.add_scalar(f"epoch_{epoch}{mini_batch_str}/parameters/{key}", value, epoch)
            else:
                self.writer.add_scalar(f"epoch_{epoch}{mini_batch_str}/parameters/{key}_mean", np.mean(value), epoch)
                self.writer.add_scalar(f"epoch_{epoch}{mini_batch_str}/parameters/{key}_std", np.std(value), epoch)
    
    def add_gradients(self, gradients, epoch, mini_batch=None):
        if not self.is_history_enabled:
            return
        """Track gradient values"""
        mini_batch_str = f'/batch_no_{mini_batch}' if mini_batch is not None else ''
        for key, value in gradients.items():
            if np.isscalar(value):
                self.writer.add_scalar(f"epoch_{epoch}{mini_batch_str}/gradients/{key}", value, epoch)
            else:
                self.writer.add_scalar(f"epoch_{epoch}{mini_batch_str}/gradients/{key}_mean", np.mean(value), epoch)
                self.writer.add_scalar(f"epoch_{epoch}{mini_batch_str}/gradients/{key}_std", np.std(value), epoch)
                self.writer.add_scalar(f"epoch_{epoch}{mini_batch_str}/gradients/{key}_norm", np.linalg.norm(value), epoch)
    
    def add_cost(self, cost, epoch, mini_batch=None):
        if not self.is_history_enabled:
            return
        """Track training cost"""
        mini_batch_str = f'/batch_no_{mini_batch}' if mini_batch is not None else ''
        self.writer.add_scalar(f"epoch_{epoch}{mini_batch_str}/cost", cost, epoch)
    
    def add_metrics(self, metrics, epoch, mini_batch=None):
        if not self.is_history_enabled:
            return
        """Track custom metrics (accuracy, etc.)"""
        mini_batch_str = f'/batch_no_{mini_batch}' if mini_batch is not None else ''
        for key, value in metrics.items():
            self.writer.add_scalar(f"epoch_{epoch}{mini_batch_str}/metrics/{key}", value, epoch)
    
    def close(self):
        """Close the writer"""
        self.writer.close()



