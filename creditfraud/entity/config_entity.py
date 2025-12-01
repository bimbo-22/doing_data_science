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
    unzip_dir: Path