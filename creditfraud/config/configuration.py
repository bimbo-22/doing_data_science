from pathlib import Path
from creditfraud.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig
)
from creditfraud.utils.common import read_yaml, create_directories
from creditfraud.constants import *

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH
    ):
        self.config = read_yaml(str(config_filepath))
        self.params = read_yaml(str(params_filepath))
        self.schema = read_yaml(str(schema_filepath))

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema
        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=Path(config.status_file),
            data_path=Path(config.data_path),
            all_schema=schema
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        return DataTransformationConfig(
            root_dir=Path(config["root_dir"]),
            data_path=Path(config["data_path"]),
            test_size=config["test_size"]
        )
