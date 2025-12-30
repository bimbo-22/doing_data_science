from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str


@dataclass
class DataValidationArtifact:
    valid_file_path: str
    invalid_file_path: str
    schema_file_path: str
    drift_report_path: str


@dataclass
class DataTransformationArtifact:
    """
    Configuration class for data transformation.
    
    Attributes:
        root_dir (Path): Root directory for data transformation artifacts.
        data_path (Path): Path to the input data file for transformation.
        test_size (float): Proportion of the dataset to include in the test split.
    """
    root_dir: Path
    data_path: Path
    test_size: float
    
@dataclass
class ModelTrainerArtifact:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    target_column: str
    
@dataclass
class ModelEvaluationArtifact:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str