import os,sys
from creditfraud.logging.logger import logging
from creditfraud.exception.exception import CreditFraudException
from creditfraud.components.data_ingestion import DataIngestion
from creditfraud.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig
)
from creditfraud.entity.artifact_entity import DataIngestionArtifact

class TrainingPipeline:
    def _init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion process.")
            print("=== data ingestion started ===")
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion process completed successfully.")
            print("=== data ingestion completed ===")
            return data_ingestion_artifact
        except Exception as e:
            raise CreditFraudException(e, sys)