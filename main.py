from src.summarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.summarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.summarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.summarizer.pipeline.stage_04_model_trainer import ModelTrainerPipeline
from src.summarizer.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

from src.summarizer.logging.logger import logger 

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> {STAGE_NAME} stage completed <<<<<<")

except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage started <<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>> {STAGE_NAME} stage completed <<<<<<")
except Exception as e:
    raise e


STAGE_NAME = "Data Transformation"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage started <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>> {STAGE_NAME} stage completed <<<<<<")
except Exception as e:
    raise e

STAGE_NAME = "Model Trainer"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage started <<<<<<")
    model_trainer = ModelTrainerPipeline()
    model_trainer.main()
    logger.info(f">>>>>> {STAGE_NAME} stage completed <<<<<<")
except Exception as e:
    raise e

STAGE_NAME = "Model Evaluation"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage started <<<<<<")
    model_eval = ModelEvaluationPipeline()
    model_eval.main()
    logger.info(f">>>>>> {STAGE_NAME} stage completed <<<<<<")
except Exception as e:
    raise e