import os 
import sys
import numpy as np
import pandas as pd

TARGET_COLUMN = "is_fraud"
PIPELINE_NAME: str = "credit_fraud_detection"
ARTIFACT_DIR: str = "Artifact"
FILE_NAME: str = "fraud_test.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"



SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

SAVED_MODEL_DIR = "saved_models"
MODEL_FILE_NAME = "model.pkl"
PREPROCESSOR_FILE_NAME = "preprocessor.pkl"

# starting with DATA INGESTION
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested_data"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.3
DATA_INGESTION_TRAIN_SPLIT_RANDOM_STATE: int = 42
DATA_INTESTION_DOWNLOAD_URL: str = "kelvinkelue/credit-card-fraud-prediction"

# DATA VALIDATION
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
VALID_FILE_NAME: str = "valid.csv"
INVALID_FILE_NAME: str = "invalid.csv"



# DATA TRANSFORMATION
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessor.pkl"

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "transformed_train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "transformed_test.npy"

# MODEL TRAINER
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD: float = 0.05