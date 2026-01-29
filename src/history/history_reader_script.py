from pathlib import Path
import yaml
import os

"""
To use this script in JupyterLab:
1. Install jupyterlab: conda install -c conda-forge jupyterlab
2. cd to the project root directory (where environments.yaml is located)
3. Run this script in JupyterLab:
    %run src/history/history_reader_script.py
4. Wait for the script to finish loading (it will print "Loaded X training records")
5. You can now use the reader object to access training history
    df.head()  # View the first 5 rows of the training history
"""
# Optional imports with error handling
try:
    from tbparse import SummaryReader
    TENSORBOARD_AVAILABLE = True
except ImportError as e:
    TENSORBOARD_AVAILABLE = False
    print(f"Warning: TensorBoard dependencies not available: {e}")

class TrainingHistoryReader:
    """Class to read training history and results."""
    
    def __init__(self, log_dir=None):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard dependencies required. Install with: pip install torch tbparse")
        
        self.log_dir = log_dir if log_dir else "logs"
        self.reader = SummaryReader(str(self.log_dir))
    
    def get_history(self):
        """Read all scalar data"""
        try:
            # Try the correct tbparse method
            return self.reader.scalars
        except AttributeError:
            try:
                return self.reader.get_scalars()
            except AttributeError:
                # Fallback: try to find the correct attribute
                available_attrs = [attr for attr in dir(self.reader) if not attr.startswith('_')]
                print(f"Available SummaryReader attributes: {available_attrs}")
                raise AttributeError(f"Cannot find scalars attribute. Available: {available_attrs}")
    
    def get_parameters(self, epoch=None):
        """Get parameter history"""
        df = self.get_history()
        if epoch:
            return df[df['tag'].str.contains(f'epoch_{epoch}/parameters/')]
        return df[df['tag'].str.contains('/parameters/')]
    
    def get_gradients(self, epoch=None):
        """Get gradient history"""
        df = self.get_history()
        if epoch:
            return df[df['tag'].str.contains(f'epoch_{epoch}/gradients/')]
        return df[df['tag'].str.contains('/gradients/')]
    
    def get_costs(self):
        """Get cost history"""
        df = self.get_history()
        return df[df['tag'].str.contains('/cost')]


def load_environments(home_dir: Path):
    """Load environment variables from YAML file"""
    try:
        env_file = home_dir / "environments.yaml"
        if env_file.exists():
            with open(env_file, "r") as f:
                environments = yaml.safe_load(f)
            
            if environments:
                # Handle nested structure
                if "default" in environments:
                    for key, value in environments["default"].items():
                        os.environ[key] = str(value)
                else:
                    # Handle flat structure
                    for key, value in environments.items():
                        os.environ[key] = str(value)
    except (FileNotFoundError, yaml.YAMLError, Exception) as e:
        print(f"Warning: Could not load environments.yaml: {e}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=str, default=None, help="Path to the home directory")
    args = parser.parse_args()
    
    # Auto-detect home directory if not provided
    home_dir = Path(args.home) if args.home else Path(__file__).parent.parent.parent
    
    # Load environments
    load_environments(home_dir)
    
    reader = TrainingHistoryReader()
    df = reader.get_history()
    print(f"Loaded {len(df)} training records")
