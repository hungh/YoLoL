
from src.configs import EnvironmentConfig
from pathlib import Path
import pytest


def get_resource_file(environment_yaml_file: str):
    test_dir = Path(__file__).parent.parent  
    return test_dir / "resources" / environment_yaml_file
    
def test_load_environments():
    config_path = get_resource_file("test_environments.yaml")
    env_config = EnvironmentConfig(config_path.relative_to(Path.cwd()))
    assert env_config.get_yolol_home() == Path("C:/Users/hungd/git/YoLoL") 
    assert env_config.get_log_dir() == Path("logs")
    assert env_config.get_history_enabled() == True
    assert env_config.get_drop_out_enabled() == True    
    assert env_config.get_gradient_checking_enabled() == False
    assert env_config.get_saved_model_dir() == Path("saved_models")

def test_gradient_checking_conflict():
    config_path = get_resource_file("test_environments2.yaml")
    env_config = EnvironmentConfig(config_path.relative_to(Path.cwd()))
    
    
    # This should raise ValueError
    with pytest.raises(ValueError, match="Dropout can not be used with gradient checking. Turn off dropout or gradient checking"):
        env_config.get_gradient_checking_enabled()

def test_sign_path():
    config_path = get_resource_file("test_environments.yaml")
    env_config = EnvironmentConfig(config_path.relative_to(Path.cwd()))
    assert env_config.get_sign_path() == Path("assets/mnist/signs")