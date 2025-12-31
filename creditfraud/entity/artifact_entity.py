from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:

    training_file_path: str
    testing_file_path: str


@dataclass
class DataValidationArtifact:
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str

    # schema_file_path: str
    # drift_report_path: str


@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_x_file_path: str
    transformed_train_y_file_path: str
    transformed_test_x_file_path: str
    transformed_test_y_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    accuracy_score: float
    precision_score: float
    recall_score: float
    roc_auc_score: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact
    