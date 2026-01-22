# from creditfraud.logging.logger import logging
# from creditfraud.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
# from creditfraud.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
# from creditfraud.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
# import pandas as pd


# STAGE_NAME = "Data Ingestion Stage"
# try:
#     logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<")
#     data_ingestion = DataIngestionTrainingPipeline()
#     data_ingestion.initiate_data_ingestion()
#     logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n\nx=====x")
# except Exception as e:
#     logging.exception(e)
#     raise e

# STAGE_NAME = "Data Validation Stage"
# try:
#     logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<")
#     data_ingestion = DataValidationTrainingPipeline()
#     data_ingestion.initiate_data_validation()
#     logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n\nx=====x")
# except Exception as e:
#     logging.exception(e)
#     raise e

# STAGE_NAME = "EDA Stage"
# try:
#     logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<")
#     df = pd.read_csv("artifacts/data_ingestion/data/fraud_test.csv")
#     logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n\nx=====x")
# except Exception as e:
#     logging.exception(e)
#     raise e

# STAGE_NAME = "Data Transformation Stage"
# try:
#     logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<")
#     data_ingestion = DataTransformationTrainingPipeline()
#     data_ingestion.initiate_data_transformation()
#     logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n\nx=====x")
# except Exception as e:
#     logging.exception(e)
#     raise e


from creditfraud.components.data_ingestion import DataIngestion
from creditfraud.entity.config_entity import DataIngestionConfig, ModelTrainerConfig,TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig
from creditfraud.components.data_validation import DataValidation
from creditfraud.components.data_transformation import DataTransformation
from creditfraud.components.model_trainer import ModelTrainer
from creditfraud.logging.logger import logging
from creditfraud.pipeline.training_pipeline import TrainingPipeline
import sys

if __name__ == "__main__":
    logging.info("==========================")
    logging.info("Initializing the configuration")
    print("=== data ingestion started ===")
    # Initialize the configuration
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    logging.info("Starting the data ingestion process")
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    logging.info("Data ingestion process completed successfully")
    print("=== data ingestion completed ===")
    
    logging.info("==========================")
    
    logging.info("Starting the data validation process")
    print("=== data validation started ===")
    data_validation_config = DataValidationConfig(training_pipeline_config)
    data_validation = DataValidation(data_validation_config, data_ingestion_artifact)
    logging.info("Data validation Initiated")
    data_validation_artifact = data_validation.initiate_data_validation()
    print(data_validation_artifact)
    logging.info("Data validation process completed successfully")
    print("=== data validation completed ===")
        
    logging.info("==========================")
    
    logging.info("Starting the data transformation process")
    print("=== data transformation started ===")
    data_transformation_config = DataTransformationConfig(training_pipeline_config)
    data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
    logging.info("Data transformation Initiated")
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    logging.info("Data transformation process completed successfully")
    print("=== data transformation completed ===")
    
    logging.info("==========================")
    
    logging.info("Starting Model Training")
    print("=== model training started ===")
    model_trainer_config =  ModelTrainerConfig(training_pipeline_config)
    model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact=data_transformation_artifact)
    model_trainer_artifact = model_trainer.initiate_model_trainer()
    logging.info("Model Traning process completed successfully")
    print("=== model training completed ===")
    logging.info("==========================")


