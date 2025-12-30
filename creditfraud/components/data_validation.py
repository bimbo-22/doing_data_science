import os, sys
import yaml
import numpy as np
import pandas as pd
import shutil
from creditfraud.exception.exception import CreditFraudException
from creditfraud.logging.logger import logging
from creditfraud.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from creditfraud.entity.config_entity import DataValidationConfig
from creditfraud.constants.training_pipeline import TARGET_COLUMN, SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact
    ):
        self.config = data_validation_config
        self.ingestion_artifact = data_ingestion_artifact

    @staticmethod
    def read_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def write_yaml(path: str, content: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(content, f, sort_keys=False)

    def validate_schema(self, df: pd.DataFrame, schema: dict) -> bool:
        schema_columns = set(schema.keys())
        data_columns = set(df.columns)

        if schema_columns != data_columns:
            logging.error("Schema column mismatch")
            return False

        if TARGET_COLUMN not in df.columns:
            logging.error("Target column missing")
            return False

        for col, meta in schema.items():
            if meta["dtype"] in ["int", "float"]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logging.error(f"Column {col} expected numeric")
                    return False

        return True

    def detect_drift(self, df: pd.DataFrame, schema: dict) -> dict:
        drift_report = {}

        for col, meta in schema.items():
            if meta["dtype"] in ["int", "float"] and not meta["is_target"]:
                drift_report[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std())
                }

        return drift_report

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")

            os.makedirs(self.config.schema_dir, exist_ok=True)
            os.makedirs(self.config.valid_dir, exist_ok=True)
            os.makedirs(self.config.invalid_dir, exist_ok=True)
            os.makedirs(self.config.drift_dir, exist_ok=True)

            df = pd.read_csv(self.ingestion_artifact.feature_store_file_path)


            if not os.path.exists(SCHEMA_FILE_PATH):
                raise FileNotFoundError("Global schema file not found")

            shutil.copy2(
                SCHEMA_FILE_PATH,
                self.config.schema_file_path
            )

            schema = self.read_yaml(self.config.schema_file_path)

            is_valid = self.validate_schema(df, schema)

            if is_valid:
                df.to_csv(self.config.valid_file_path, index=False)
            else:
                df.to_csv(self.config.invalid_file_path, index=False)

            drift_report = self.detect_drift(df, schema)
            self.write_yaml(self.config.drift_report_path, drift_report)

            return DataValidationArtifact(
                valid_file_path=self.config.valid_file_path,
                invalid_file_path=self.config.invalid_file_path,
                schema_file_path=self.config.schema_file_path,
                drift_report_path=self.config.drift_report_path
            )

        except Exception as e:
            raise CreditFraudException(e, sys)
