import os
import sys
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import shap
from mlflow.models.signature import infer_signature

from scipy.sparse import load_npz
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from creditfraud.exception.exception import CreditFraudException
from creditfraud.logging.logger import logging
from creditfraud.entity.config_entity import ModelTrainerConfig
from creditfraud.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from creditfraud.utils.main_utils.utils import (
    save_object,
    load_object,
    evaluate_models,
)
from creditfraud.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)
from creditfraud.utils.ml_utils.model.estimator import FraudModel


# ============================================================
# MLflow SAFE LOCAL SETUP (cross-OS, non-root)
# ============================================================
PROJECT_ROOT = os.getcwd()
MLFLOW_DIR = os.path.join(PROJECT_ROOT, "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.replace(os.sep, '/')}")
mlflow.set_experiment("credit_fraud_detection")


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    # ============================================================
    # REQUIRED METHODS (YOU CALLED OUT — NOW INCLUDED)
    # ============================================================
    def tune_threshold(self, y_true, y_proba):
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx]

    def precision_at_k(self, y_true, y_proba, k):
        idx = np.argsort(y_proba)[::-1][:k]
        return y_true[idx].sum() / k

    def log_feature_importance(self, model):
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            plt.figure(figsize=(10, 4))
            plt.plot(np.sort(fi)[::-1][:50])
            plt.title("Top Feature Importances")
            mlflow.log_figure(plt.gcf(), "feature_importance.png")
            plt.close()

    def log_shap(self, model, X_sample):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap.summary_plot(shap_values, X_sample, show=False)
            mlflow.log_figure(plt.gcf(), "shap_summary.png")
            plt.close()
        except Exception as e:
            logging.warning(f"SHAP skipped: {e}")

    # ============================================================
    # TRAINING LOGIC
    # ============================================================
    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, class_weight="balanced", n_jobs=-1
            ),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "Xgboost": XGBClassifier(eval_metric="logloss"),
            "LightGBM": LGBMClassifier(class_weight="balanced", verbose=-1),
        }

        # ========================================================
        # YOUR HYPERPARAMETERS (UNCHANGED, COMMENTS KEPT)
        # ========================================================
        params = {
            "DecisionTreeClassifier": {
                "criterion": ["gini", "entropy", "log_loss"],
                "splitter": ["best", "random"],
                "max_depth": [1, 2, 3, 4, 5],
                "max_features": ["auto", "sqrt", "log2"],
            },
            "LightGBM": {
                "num_leaves": [31, 50, 70],
                "learning_rate": [0.1, 0.01, 0.05],
                "n_estimators": [100, 200, 500],
            },
            "RandomForestClassifier": {
                "max_depth": [5, 8, 15, None, 10],
                "max_features": [5, 7, 8],
                "min_samples_split": [2, 8, 15, 20],
                "n_estimators": [100, 200, 500, 1000],
            },
            "GradientBoostingClassifier": {
                "loss": ["log_loss", "exponential"],
                "criterion": ["friedman_mse", "squared_error", "mse"],
                "min_samples_split": [2, 8, 15, 20],
                "n_estimators": [100, 200, 500],
                "max_depth": [5, 8, 15, None, 10],
            },
            "LogisticRegression": {
                "penalty": ["l2", None],
                # "C": [100, 10, 1.0, 0.1, 0.01],
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            },
            "AdaBoostClassifier": {
                "n_estimators": [50, 60, 70, 80, 90],
                "algorithm": ["SAMME"],
            },
            "Xgboost": {
                "learning_rate": [0.1, 0.01],
                "max_depth": [5, 8, 12, 20, 30],
                "n_estimators": [100, 200, 300],
                "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4],
            },
        }

        model_report, best_params_all, all_metrics = evaluate_models(
            X_train, y_train, X_test, y_test, models, params
        )

        best_model_name = max(model_report, key=model_report.get)
        best_model = models[best_model_name]
        logging.info(f"Best model selected: {best_model_name}")

        best_model.fit(X_train, y_train)

        y_train_proba = best_model.predict_proba(X_train)[:, 1]
        y_test_proba = best_model.predict_proba(X_test)[:, 1]

        best_threshold = self.tune_threshold(y_train, y_train_proba)

        y_train_pred = (y_train_proba >= best_threshold).astype(int)
        y_test_pred = (y_test_proba >= best_threshold).astype(int)

        train_metric = get_classification_score(y_train, y_train_pred)
        test_metric = get_classification_score(y_test, y_test_pred)

        # ========================================================
        # MLflow logging
        # ========================================================
        with mlflow.start_run(run_name=best_model_name):

            mlflow.log_param("best_model", best_model_name)
            mlflow.log_param("threshold", float(best_threshold))
            mlflow.log_params(best_params_all.get(best_model_name, {}))

            mlflow.log_metrics({
                "train_f1": train_metric.f1_score,
                "train_accuracy": train_metric.accuracy_score,
                "train_precision": train_metric.precision_score,
                "train_recall": train_metric.recall_score,
                "train_roc_auc": train_metric.roc_auc_score,

                "test_f1": test_metric.f1_score,
                "test_accuracy": test_metric.accuracy_score,
                "test_precision": test_metric.precision_score,
                "test_recall": test_metric.recall_score,
                "test_roc_auc": test_metric.roc_auc_score,
            })

            mlflow.log_metric(
                "precision_at_100",
                self.precision_at_k(y_test, y_test_proba, 100),
            )

            # Curves
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            plt.figure()
            plt.plot(fpr, tpr)
            mlflow.log_figure(plt.gcf(), "roc_curve.png")
            plt.close()

            precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
            plt.figure()
            plt.plot(recall, precision)
            mlflow.log_figure(plt.gcf(), "precision_recall_curve.png")
            plt.close()

            cm = confusion_matrix(y_test, y_test_pred)
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(ax=ax)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            self.log_feature_importance(best_model)

            X_sample = (
                X_train[:1000].toarray()
                if hasattr(X_train, "toarray")
                else X_train[:1000]
            )
            self.log_shap(best_model, X_sample)
            input_example = X_train[:5].toarray()  # Convert sparse to dense for example
            predictions = best_model.predict(input_example)
            signature = infer_signature(input_example, predictions)
            mlflow.sklearn.log_model(
                best_model,
                name="model",
                registered_model_name="CreditFraudModel",
                signature=signature,
                input_example=input_example,
            )

        # ========================================================
        # SAVE FraudModel (model + preprocessor + threshold)
        # ========================================================
        preprocessor = load_object(
            self.data_transformation_artifact.transformed_object_file_path
        )
        

        fraud_model = FraudModel(
            preprocessor=preprocessor,
            model=best_model,
        )
        fraud_model.threshold = best_threshold

        os.makedirs(
            os.path.dirname(self.model_trainer_config.trained_model_file_path),
            exist_ok=True,
        )

        save_object(
            self.model_trainer_config.trained_model_file_path,
            fraud_model,
        )

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric,
        )

    # ============================================================
    # PIPELINE ENTRY POINT
    # ============================================================
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            X_train = load_npz(
                self.data_transformation_artifact.transformed_train_x_file_path
            )
            y_train = np.load(
                self.data_transformation_artifact.transformed_train_y_file_path
            )
            X_test = load_npz(
                self.data_transformation_artifact.transformed_test_x_file_path
            )
            y_test = np.load(
                self.data_transformation_artifact.transformed_test_y_file_path
            )

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise CreditFraudException(e, sys)
