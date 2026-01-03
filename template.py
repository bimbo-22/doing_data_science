import os 
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

project_name="creditfraud"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/components/data_ingestion.py",
    f"{project_name}/components/data_transformation.py",
    f"{project_name}/components/model_trainer.py",
    f"{project_name}/components/model_evaluation.py",
    f"{project_name}/components/data_validation.py",
    f"{project_name}/components/feature_extractor.py",
    f"{project_name}/logging/__init__.py",
    f"{project_name}/logging/logger.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/utils/common.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/entity/__init___.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/constants/training_pipeline.py/__init__.py",
    f"{project_name}/constants/__init__.py",\
    "config/config.yaml",
    "params.yaml",
    "main.py",
    "schema.yaml",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/research.ipynb",

    
    
    
]

for file in list_of_files:
    file_path = Path(file)
    filedir, file_name = os.path.split(file_path)
    
    if filedir!="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {file_name}")
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path, 'w') as f:
            pass
        logging.info(f"Creating empty file: {file_name}")
    
    else:
        logging.info(f"File already exists: {file_name}")
        
        
        
        import os, sys
import numpy as np
import mlflow
import yaml
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from creditfraud.logging.logger import logging
from creditfraud.exception.exception import CreditFraudException
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
from creditfraud.utils.ml_utils.model.estimator import FraudModel
from creditfraud.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
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
from sklearn.metrics import precision_recall_curve
from mlflow.models import infer_signature

# MLflow env
MLFLOW_DIR = os.path.join(os.getcwd(), "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.replace(os.sep, '/')}")

class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

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
    
    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000, class_weight="balanced", n_jobs=-1
            ),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "Xgboost": XGBClassifier(
                eval_metric="logloss"  
            ),
            "LightGBM": LGBMClassifier(class_weight="balanced", verbose=-1),  # Suppress LightGBM info logs
        }

        params = {
            "DecisionTreeClassifier": {
                "criterion": ["gini", "entropy", "log_loss"],
                # "splitter": ["best", "random"],
                # "max_depth": [1, 2, 3, 4, 5],
                # "max_features": ["auto", "sqrt", "log2"],
            },
            "LightGBM": {
                "num_leaves": [31, 50, 70],
                # "learning_rate": [0.1, 0.01, 0.05],
                # "n_estimators": [100, 200, 500],
            },
            "RandomForestClassifier": {
                "max_depth": [5, 8, 15, None, 10],
                # "max_features": [5, 7, 8],
                # "min_samples_split": [2, 8, 15, 20],
                # "n_estimators": [100, 200, 500, 1000],
            },
            "GradientBoostingClassifier": {
                "loss": ['log_loss','exponential'],
                    # "criterion": ['friedman_mse','squared_error','mse'],
                    # "min_samples_split": [2, 8, 15, 20],
                    # "n_estimators": [100, 200, 500],
                    # "max_depth": [5, 8, 15, None, 10]
                        },
                "LogisticRegression": {
                    'penalty': ['l2', None],
                    # 'C': [100,10,1.0,0.1,0.01],
                    # 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                },
                    
                "AdaBoostClassifier": {
                "n_estimators":[50,60,70,80,90],
                # "algorithm":['SAMME']
                },
                "Xgboost":{"learning_rate": [0.1, 0.01],
                #   "max_depth": [5, 8, 12, 20, 30],
                #   "n_estimators": [100, 200, 300],
                #   "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]}
                }
                  
        }
        

        model_report, best_params_all, all_metrics = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            params=params,
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

        metrics_path = os.path.join(
            os.path.dirname(self.model_trainer_config.trained_model_file_path),
            "metrics.yaml"
        )
        os.makedirs(
            os.path.dirname(self.model_trainer_config.trained_model_file_path),
            exist_ok=True,
        )
        metrics = {
            "all_models": {
                model_name: {
                    "hyperparameters": best_params_all.get(model_name, {}),
                    "train": {k: float(v) for k, v in all_metrics[model_name]["train"].items()},
                    "test": {k: float(v) for k, v in all_metrics[model_name]["test"].items()},
                } for model_name in model_report
            },
            "best_model": {
                "name": best_model_name,
                "threshold": float(best_threshold),
                "train": {
                    "f1_score": float(train_metric.f1_score),
                    "accuracy_score": float(train_metric.accuracy_score),
                    "precision_score": float(train_metric.precision_score),
                    "recall_score": float(train_metric.recall_score),
                    "roc_auc_score": float(train_metric.roc_auc_score),
                },
                "test": {
                    "f1_score": float(test_metric.f1_score),
                    "accuracy_score": float(test_metric.accuracy_score),
                    "precision_score": float(test_metric.precision_score),
                    "recall_score": float(test_metric.recall_score),
                    "roc_auc_score": float(test_metric.roc_auc_score),
                },
            }
        }
        with open(metrics_path, 'w') as f:
            yaml.safe_dump(metrics, f)
        logging.info(f"Metrics saved to {metrics_path}")
        mlflow.set_tracking_uri(f"file:///{MLFLOW_DIR.replace(os.sep, '/')}")
        with mlflow.start_run(run_name=best_model_name):
            # Log best hyperparameters
            best_params = best_params_all.get(best_model_name, {})
            mlflow.log_params(best_params)
            mlflow.log_params({"model": best_model_name, "threshold": best_threshold})
            # Log metrics
            mlflow.log_metrics(
                {
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
                }
            )
            # Generate and log plots using mlflow.log_figure for direct tracking in MLflow
            # ROC Curve: Shows the trade-off between true positive rate and false positive rate; higher AUC indicates better model performance
            fpr, tpr, _ = roc_curve(y_test, y_test_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close(fig)

            # Precision-Recall Curve: Useful for imbalanced datasets like fraud; shows precision vs. recall trade-off
            precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, color='blue', lw=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            mlflow.log_figure(fig, "pr_curve.png")
            plt.close(fig)

            # Confusion Matrix: Visualizes correct and incorrect predictions; helps identify false positives/negatives
            cm = confusion_matrix(y_test, y_test_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap=plt.cm.Blues)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            # Feature Importance (if model supports it): Shows which features contribute most to predictions; useful for interpretability
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                # Assuming you have feature names; otherwise, use indices
                feature_names = [f"feature_{i}" for i in range(len(importances))]  # Placeholder
                sorted_idx = np.argsort(importances)[::-1]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(importances)), importances[sorted_idx], align='center')
                ax.set_xticks(range(len(importances)))
                ax.set_xticklabels(np.array(feature_names)[sorted_idx], rotation=90)
                ax.set_title('Feature Importances')
                mlflow.log_figure(fig, "feature_importance.png")
                plt.close(fig)

            # Infer signature and use input_example to avoid warnings
            input_example = X_train[:5].toarray()  # Convert sparse to dense for example
            predictions = best_model.predict(input_example)
            signature = infer_signature(input_example, predictions)
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",  # Use 'name' instead of deprecated 'artifact_path'
                signature=signature,
                input_example=input_example
            )

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
        save_object(self.model_trainer_config.trained_model_file_path, fraud_model)
        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric,
        )

 
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
    