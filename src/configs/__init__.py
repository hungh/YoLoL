import os
from pathlib import Path
import yaml

# path: src.configs.EnvironmentConfig

class EnvironmentConfig:
    def __init__(self, config_file_path: Path = Path("environments.yaml")):
        self.config_file_path = config_file_path
        self.load_environments(Path.cwd())
        self.yolol_home = Path(os.environ.get('YOLOL_HOME', Path.cwd()))                
        self.history_enabled = os.environ.get('HISTORY_ENABLED', 'false').lower().strip() == 'true'
        self.drop_out_enabled = os.environ.get('DROP_OUT_ENABLED', 'false').lower().strip() == 'true'
        self.gradient_checking_enabled = os.environ.get('GRADIENT_CHECKING_ENABLED', 'false').lower().strip() == 'true'        
        self.log_dir = Path(os.environ.get('LOG_DIR', Path.cwd() / "logs"))        

    def get_yolol_home(self) -> Path:        
        return self.yolol_home

    def get_history_enabled(self) -> bool:        
        return self.history_enabled
    
    def get_drop_out_enabled(self) -> bool:
        return self.drop_out_enabled
    
    def get_log_dir(self) -> Path:
        print(f"Log dir: {self.log_dir}")
        return self.log_dir

    def get_sign_path(self) -> Path:
        return Path(os.environ.get('mnst.SIGN_PATH', Path.cwd() / "assets/mnist/signs"))
    
    def get_saved_model_dir(self) -> Path:
        return Path(os.environ.get('SAVED_MODEL_DIR', Path.cwd() / "saved_model"))

    def get_gradient_checking_enabled(self) -> bool:
        gradient_checking_enabled = self.gradient_checking_enabled
        if self.gradient_checking_enabled and self.drop_out_enabled:
            print("Dropout can not be used with gradient checking")
            raise ValueError("Dropout can not be used with gradient checking. Turn off dropout or gradient checking")

        return gradient_checking_enabled

    def load_environments(self, home_dir: Path):
        """Load environment variables from YAML file"""
        print(f"Loading environment variables from {self.config_file_path}")
        try:
            env_file = home_dir / self.config_file_path
            print(f"Environment file: {env_file} exists: {env_file.exists()}")
            if env_file.exists():
                with open(env_file, "r") as f:
                    environments = yaml.safe_load(f)
                
                if environments:
                    # Handle nested structure
                    if "default" in environments:
                        for key, value in environments["default"].items():
                            os.environ[key] = str(value)
                    # load environment with prefix
                    for env_name, env_values in environments.items():
                        if env_name != "default":
                            for key, value in env_values.items():
                                os.environ[f'{env_name}.{key}'] = str(value)
        except (FileNotFoundError, yaml.YAMLError, Exception) as e:
            print(f"Warning: Could not load {self.config_file_path}: {e}")