import os
from pathlib import Path

class EnvironmentConfig:
    def __init__(self):
        self.yolol_home = os.environ.get('YOLOL_HOME', Path.cwd())

    def get_yolol_home(self):
        return self.yolol_home
