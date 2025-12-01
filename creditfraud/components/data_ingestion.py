import os
import shutil
import kagglehub
from creditfraud.entity.config_entity import DataIngestionConfig
from creditfraud.logging.logger import logging


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        target_dir = self.config.local_data_file   


        if os.path.exists(target_dir):
            logging.info(f"Dataset already exists at {target_dir}. Skipping download.")
            return

        logging.info(f"Downloading dataset from Kaggle: {self.config.source_URL}")
        
        # Download dataset through KaggleHub
        kaggle_path = kagglehub.dataset_download(self.config.source_URL)
        logging.info(f"Dataset downloaded to temporary directory: {kaggle_path}")

        # Copy dataset folder into artifacts
        shutil.copytree(kaggle_path, target_dir)
        logging.info(f"Dataset successfully copied to: {target_dir}")

        self.rename_to_fraud_test()

    def rename_to_fraud_test(self):
        data_dir = self.config.local_data_file

        if not os.path.exists(data_dir):
            logging.warning(f"Data directory not found: {data_dir}")
            return

        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

        if not csv_files:
            logging.warning(f"No CSV files found in {data_dir}. Cannot rename.")
            return

        original_path = os.path.join(data_dir, csv_files[0])
        new_path = os.path.join(data_dir, "fraud_test.csv")

        if os.path.exists(new_path):
            os.remove(new_path)

        os.rename(original_path, new_path)
        logging.info(f"Renamed {csv_files[0]} → fraud_test.csv")

