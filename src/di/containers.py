from dependency_injector import containers, providers
from ..configs import EnvironmentConfig
from ..history import TrainingHistoryWriter
from .providers import create_trainer_providers

class  Container(containers.DeclarativeContainer):
    env_config = providers.Singleton(EnvironmentConfig)

    history_writer = providers.Factory(
        TrainingHistoryWriter,
        log_dir=env_config.provided.get_log_dir.call(), 
        env_config=env_config
    )

    mini_batch_trainer, mini_batch_trainer_with_adam = create_trainer_providers(history_writer)
    