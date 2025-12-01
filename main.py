from creditfraud.logging.logger import logging
from creditfraud.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from creditfraud.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from creditfraud.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
import pandas as pd


STAGE_NAME = "Data Ingestion Stage"
try:
    logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n\nx=====x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<")
    data_ingestion = DataValidationTrainingPipeline()
    data_ingestion.initiate_data_validation()
    logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n\nx=====x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "EDA Stage"
try:
    logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<")
    df = pd.read_csv("artifacts/data_ingestion/data/fraud_test.csv")
    logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n\nx=====x")
except Exception as e:
    logging.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<")
    data_ingestion = DataTransformationTrainingPipeline()
    data_ingestion.initiate_data_transformation()
    logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n\nx=====x")
except Exception as e:
    logging.exception(e)
    raise e