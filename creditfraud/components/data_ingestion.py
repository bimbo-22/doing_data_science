import os, sys
import shutil
import kagglehub
import pandas as pd
import yaml

from creditfraud.entity.config_entity import DataIngestionConfig
from creditfraud.exception.exception import CreditFraudException
from creditfraud.entity.artifact_entity import DataIngestionArtifact
from creditfraud.logging.logger import logging
from creditfraud.constants.training_pipeline import (
    TARGET_COLUMN,
    SCHEMA_FILE_PATH
)
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CreditFraudException(e, sys)

    @staticmethod
    def infer_schema(df: pd.DataFrame) -> dict:
        """
        Infer schema (dtype, nullable, target) from dataframe
        """
        schema = {}

        for column in df.columns:
            dtype = df[column].dtype

            if pd.api.types.is_integer_dtype(dtype):
                col_type = "int"
            elif pd.api.types.is_float_dtype(dtype):
                col_type = "float"
            elif pd.api.types.is_bool_dtype(dtype):
                col_type = "bool"
            else:
                col_type = "categorical"

            schema[column] = {
                "dtype": col_type,
                "nullable": bool(df[column].isnull().any()),
                "is_target": column == TARGET_COLUMN
            }

        return schema

    @staticmethod
    def save_schema(schema: dict, schema_path: str):
        os.makedirs(os.path.dirname(schema_path), exist_ok=True)
        with open(schema_path, "w") as f:
            yaml.dump(schema, f, sort_keys=False)

    def download_file_(self):
        try:
            file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            if os.path.exists(file_path):
                logging.info("Feature store file already exists. Skipping download.")
                return

            kaggle_path = kagglehub.dataset_download(
                self.data_ingestion_config.source_url
            )

            csv_files = [
                f for f in os.listdir(kaggle_path) if f.endswith(".csv")
            ]

            if not csv_files:
                raise CreditFraudException(
                    Exception("No CSV file found in Kaggle dataset"), sys
                )

            shutil.copy2(
                os.path.join(kaggle_path, csv_files[0]),
                file_path
            )

            logging.info(f"Dataset copied to {file_path}")

            # get the schema
            df = pd.read_csv(file_path)
            schema = self.infer_schema(df)
            self.save_schema(schema, SCHEMA_FILE_PATH)

            logging.info(f"Schema saved to {SCHEMA_FILE_PATH}")
            return df

        except Exception as e:
            raise CreditFraudException(e, sys)
    def split_data_into_train_test(self, dataframe: pd.DataFrame):
        try: 
            train_set, test_set = train_test_split(
                dataframe, 
                test_size=self.data_ingestion_config.train_test_split_ratio, 
                random_state=self.data_ingestion_config.train_test_split_random_state,
                stratify=dataframe[TARGET_COLUMN])
            logging.info("performing train test split on the dataframe")
            
            logging.info("train test split completed")
            
            dir_path =  os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info("exporting train and test file path ")
            
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            
            logging.info("train and test file path exported")
            
        except Exception as e:
            raise CreditFraudException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            dataframe = self.download_file_()
            self.split_data_into_train_test(dataframe)
            logging.info("Data ingestion completed successfully.")

            return DataIngestionArtifact(
                training_file_path=self.data_ingestion_config.training_file_path,
                testing_file_path=self.data_ingestion_config.testing_file_path
            )

        except Exception as e:
            raise CreditFraudException(e, sys)
