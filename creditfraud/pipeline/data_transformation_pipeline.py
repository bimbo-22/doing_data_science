import mlflow
from pathlib import Path
from creditfraud.config.configuration import ConfigurationManager
from creditfraud.components.data_transformation import DataTransformation
from creditfraud.logging.logger import logging

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_transformation(self):
        logging.info(f">>>>>> Starting {STAGE_NAME} <<<<<<")

        try:
            status_file = Path("artifacts/data_validation/status.txt")
            with open(status_file, "r") as f:
                status = f.read().split(" ")[-1].strip()

            if status != "True":
                logging.info("Data schema is not valid. Transformation halted.")
                return

            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()

            mlflow.set_experiment("FraudDetection_Transformation")

            with mlflow.start_run(run_name="data_transformation"):
                logging.info("MLflow run started for Data Transformation.")
                mlflow.log_param("test_size", data_transformation_config.test_size)

                data_transformation = DataTransformation(
                    config=data_transformation_config
                )

                cleaned_df, train_df, test_df = data_transformation.run_transformation()

                # Log artifacts
                mlflow.log_artifact(str(data_transformation_config.root_dir / "cleaned.csv"))
                mlflow.log_artifact(str(data_transformation_config.root_dir / "train.csv"))
                mlflow.log_artifact(str(data_transformation_config.root_dir / "test.csv"))

                # Metrics
                mlflow.log_metric("clean_rows", cleaned_df.shape[0])
                mlflow.log_metric("train_rows", train_df.shape[0])
                mlflow.log_metric("test_rows", test_df.shape[0])

                mlflow.log_metric("fraud_rate_train", train_df["is_fraud"].mean())
                mlflow.log_metric("fraud_rate_test", test_df["is_fraud"].mean())

                logging.info("Data Transformation completed successfully.")
                logging.info(f"Train shape: {train_df.shape}")
                logging.info(f"Test shape: {test_df.shape}")

            logging.info(f">>>>>> Completed {STAGE_NAME} <<<<<<")

        except Exception as e:
            logging.exception(e)
            raise e
