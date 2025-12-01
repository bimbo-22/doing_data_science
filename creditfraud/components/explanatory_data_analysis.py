import pandas as pd
import mlflow

class FraudEDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        mlflow.set_experiment("fraud_detection")

    def run_eda(self):
        with mlflow.start_run(run_name="eda_run"):

            # Log basic dataset shape
            mlflow.log_metric("rows", self.df.shape[0])
            mlflow.log_metric("columns", self.df.shape[1])

            # Missing values
            total_missing = self.df.isnull().sum().sum()
            mlflow.log_metric("total_missing_values", total_missing)

            # Fraud stats
            fraud_count = (self.df["is_fraud"] == 1).sum()
            mlflow.log_metric("fraud_count", fraud_count)
            mlflow.log_metric("fraud_percentage", fraud_count / len(self.df) * 100)

            # Merchant stats
            mlflow.log_metric("unique_merchants", self.df["merchant"].nunique())

            # Save descriptive statistics as an artifact
            describe_path = "eda_describe.csv"
            self.df.describe().to_csv(describe_path)
            mlflow.log_artifact(describe_path)

            # Save missing value breakdown
            missing_path = "eda_missing.csv"
            self.df.isnull().sum().to_csv(missing_path)
            mlflow.log_artifact(missing_path)

            print("\nEDA tracking complete. Saved to MLflow.")
