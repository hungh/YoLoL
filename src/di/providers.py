from dependency_injector import providers
from ..classic_train.mini_batch.trainer import MiniBatchTrainer
from ..classic_train.mini_batch.trainer_with_adam import TrainerWithAdam

def create_trainer_providers(history_writer):    
    mini_batch_trainer = providers.Factory(
        MiniBatchTrainer,
        history_writer=history_writer
    )
    
    mini_batch_trainer_with_adam = providers.Factory(
        TrainerWithAdam,
        history_writer=history_writer
    )
    
    return mini_batch_trainer, mini_batch_trainer_with_adam
