from src.textSummarizer.pipeline.stage_01 import DataIngestionPipeline
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