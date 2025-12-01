from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    
    Attributes:
        source_url (str): URL of the source data.
        local_data_file (Path): Path to the local data file.
        unzip_dir (Path): Directory to unzip the data.
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass
class DataValidationConfig:
    """
    Configuration class for data validation.
    
    Attributes:
        root_dir (Path): Root directory for data validation.
        STATUS_FILE (Path): Path to the status file.
        unzip_data_dir (Path): Directory where the data is unzipped.
        all_schema (dict): Schema for data validation.
    """
    root_dir: Path
    STATUS_FILE: Path
    all_schema: dict
    data_path: Path


@dataclass
class DataTransformationConfig:
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