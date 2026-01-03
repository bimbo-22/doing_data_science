import sys
import pandas as pd
import numpy as np
from scipy.sparse import save_npz
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import category_encoders as ce
from creditfraud.entity.config_entity import DataTransformationConfig
from creditfraud.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from creditfraud.exception.exception import CreditFraudException
from creditfraud.logging.logger import logging
from creditfraud.constants.training_pipeline import TARGET_COLUMN
from creditfraud.utils.main_utils.utils import *


class DataTransformation:
    def __init__(self, data_validation_artifact:DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    @staticmethod
    def read_data(file_path):
        return pd.read_csv(file_path)

    def apply_transformations(self, df):
        if "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "id"}, inplace=True)

        df["merchant"] = df["merchant"].str.replace("fraud_", "", regex=False)

        df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
        )

        df["year"] = df["trans_date_trans_time"].dt.year
        df["month"] = df["trans_date_trans_time"].dt.month
        df["day"] = df["trans_date_trans_time"].dt.day
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df["dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
        df.drop(columns=["trans_date_trans_time"], inplace=True)

        df.drop(
            columns=[
                "cc_num", "dob", "first", "last", "street",
                "trans_num", "id", "zip", "unix_time", "lat", "long"
            ],
            errors="ignore",
            inplace=True,
        )

        return df

    def get_preprocessor(self, target_enc_cols, one_hot_cols, numeric_cols):
        return ColumnTransformer(
            transformers=[
                ("target_enc", ce.TargetEncoder(), target_enc_cols),
                ("onehot", OneHotEncoder(handle_unknown="ignore"), one_hot_cols),
                ("num", StandardScaler(), numeric_cols),
            ]
        )
    
    def initiate_data_transformation(self):
        try:
            train_df = self.apply_transformations(
                self.read_data(self.data_validation_artifact.valid_train_file_path)
            )
            test_df = self.apply_transformations(
                self.read_data(self.data_validation_artifact.valid_test_file_path)
            )

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN].to_numpy()

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN].to_numpy()

            cat_cols = X_train.select_dtypes(include="object").columns.tolist()
            num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

            preprocessor = self.get_preprocessor(
                target_enc_cols=[],
                one_hot_cols=cat_cols,
                numeric_cols=num_cols,
            )

            X_train_t = preprocessor.fit_transform(X_train, y_train)
            X_test_t = preprocessor.transform(X_test)

            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)

            save_npz(self.data_transformation_config.transformed_train_x_file_path, X_train_t)
            save_npz(self.data_transformation_config.transformed_test_x_file_path, X_test_t)
            np.save(self.data_transformation_config.transformed_train_y_file_path, y_train)
            np.save(self.data_transformation_config.transformed_test_y_file_path, y_test)

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_x_file_path=self.data_transformation_config.transformed_train_x_file_path,
                transformed_train_y_file_path=self.data_transformation_config.transformed_train_y_file_path,
                transformed_test_x_file_path=self.data_transformation_config.transformed_test_x_file_path,
                transformed_test_y_file_path=self.data_transformation_config.transformed_test_y_file_path,
            )

        except Exception as e:
            raise CreditFraudException(e, sys)
