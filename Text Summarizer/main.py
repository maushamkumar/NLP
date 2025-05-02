from src.textSummarizer.pipeline.stage_01 import DataIngestionPipeline
from src.textSummarizer.pipeline.stage_02 import DataValidationPipeline
from src.textSummarizer.pipeline.stage_03 import DataTransformationPipeline
from src.textSummarizer.pipeline.stage_o4_model_trainign import ModelTrainingPipeline
from src.textSummarizer.logging import logger


STAGE_NAME = "Data Ingestion stage"
try: 
    logger.info(f'========================={STAGE_NAME} started=========================')
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info(f'========================={STAGE_NAME} completed=========================')
except Exception as e:
    logger.exception(f"An error occurred: {e}")
    raise e

STAGE_NAME = "Data Validation "
try: 
    logger.info(f'========================={STAGE_NAME} started=========================')
    data_validation = DataValidationPipeline()
    data_validation.main()
    logger.info(f'========================={STAGE_NAME} completed=========================')
except Exception as e:
    logger.exception(f"An error occurred: {e}")
    raise e


STAGE_NAME = "Data Transformation "
try: 
    logger.info(f'========================={STAGE_NAME} started=========================')
    data_validation = DataTransformationPipeline()
    data_validation.main()
    logger.info(f'========================={STAGE_NAME} completed=========================')
except Exception as e:
    logger.exception(f"An error occurred: {e}")
    raise e


STAGE_NAME = " Model Training "
try: 
    logger.info(f'========================={STAGE_NAME} started=========================')
    data_validation = ModelTrainingPipeline()
    data_validation.main()
    logger.info(f'========================={STAGE_NAME} completed=========================')
except Exception as e:
    logger.exception(f"An error occurred: {e}")
    raise e