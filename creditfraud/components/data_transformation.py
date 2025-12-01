import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from creditfraud.entity.config_entity import DataTransformationConfig
from creditfraud.logging.logger import logging

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.config.root_dir.mkdir(parents=True, exist_ok=True)

    def apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting transformation...")

        df = df.copy()

        # Rename variables
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "id"})

        # Remove fraud_ prefix
        df["merchant"] = df["merchant"].str.replace("fraud_", "", regex=False)

        # Split date & time
        df[["trans_date", "trans_time"]] = df["trans_date_trans_time"].str.split(" ", expand=True)
        df = df.drop(columns=["trans_date_trans_time"])

        return df

    def split_and_save(self, df: pd.DataFrame):
        logging.info("Starting train-test split...")

        train, test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=42,
            stratify=df["is_fraud"]
        )

        # Paths
        cleaned_path = self.config.root_dir / "cleaned.csv"
        train_path = self.config.root_dir / "train.csv"
        test_path = self.config.root_dir / "test.csv"

        # Save all CSVs
        df.to_csv(cleaned_path, index=False)
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        logging.info(f"Saved cleaned dataset: {cleaned_path}")
        logging.info(f"Saved train dataset: {train_path}")
        logging.info(f"Saved test dataset: {test_path}")

        return df, train, test

    def run_transformation(self):
        logging.info("Loading data...")

        df = pd.read_csv(self.config.data_path)
        logging.info(f"Raw data shape: {df.shape}")

        # Apply transformations
        df_clean = self.apply_transformations(df)
        logging.info(f"Cleaned data shape: {df_clean.shape}")

        # Split + save
        cleaned, train, test = self.split_and_save(df_clean)

        return cleaned, train, test
