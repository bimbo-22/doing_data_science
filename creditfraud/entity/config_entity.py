from datetime import datetime
from dataclasses import dataclass
import os
from creditfraud.constants import training_pipeline

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m-%d-%Y-%H-%M-%S")
        self.pipeline_name =  training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp: str = timestamp
        self.model_dir = os.path.join("final_model")
        
class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME,
        )
        self.source_url = training_pipeline.DATA_INTESTION_DOWNLOAD_URL
        
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.train_test_split_random_state: int = training_pipeline.DATA_INGESTION_TRAIN_SPLIT_RANDOM_STATE

        
class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )

        # Schema (versioned per run)
        self.schema_dir = os.path.join(self.data_validation_dir, "schema")
        self.schema_file_path = os.path.join(
            self.schema_dir,
            "schema.yaml"
        )

        # # Drift
        # self.drift_dir = os.path.join(self.data_validation_dir, "drift")
        # self.drift_report_path = os.path.join(
        #     self.drift_dir,
        #     "drift_report.yaml"
        # )

        # Valid / Invalid data
        self.valid_dir = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )

        self.invalid_dir = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )

        self.valid_train_file_path = os.path.join(
            self.valid_dir,
            training_pipeline.TRAIN_FILE_NAME
        )

        self.invalid_train_file_path = os.path.join(
            self.invalid_dir,
            training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path = os.path.join(
            self.valid_dir,
            training_pipeline.TEST_FILE_NAME
        )
        self.invalid_test_file_path = os.path.join(
            self.valid_dir,
            training_pipeline.TEST_FILE_NAME
        )
    

            
class DataTransformationConfig:
    def __init__(self, training_pipeline_config):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )

        self.transformed_data_dir = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
        )

        self.transformed_object_file_path = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
        )
        self.resampling_method = 'smotetomek'  # Options: 'smotetomek', 'none' , etc.

        self.transformed_train_x_file_path = os.path.join(
            self.transformed_data_dir, "train_X.npz"
        )
        self.transformed_train_y_file_path = os.path.join(
            self.transformed_data_dir, "train_y.npy"
        )
        self.transformed_test_x_file_path = os.path.join(
            self.transformed_data_dir, "test_X.npz"
        )
        self.transformed_test_y_file_path = os.path.join(
            self.transformed_data_dir, "test_y.npy"
        )

        
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,training_pipeline.MODEL_TRAINER_TRAINED_MODEL_NAME
        )
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD
        