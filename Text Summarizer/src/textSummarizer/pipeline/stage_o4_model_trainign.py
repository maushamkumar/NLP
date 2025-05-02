from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.model_training import ModelTrainer
from src.textSummarizer.logging import logger


class ModelTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_training_config()
            model_trainer_config = ModelTrainer(config=model_trainer_config)
            model_trainer_config.train()
        except Exception as e:
            # logger.exception(f"An error occurred: {e}")
            raise e
