from creditfraud.components.data_ingestion import DataIngestion
from creditfraud.config.configuration import ConfigurationManager
from creditfraud.logging.logger import logging 

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
            config = ConfigurationManager() # from componennts/configuration.py
            data_ingestion_config = config.get_data_ingestion_config() # from componennts/configuration.py
            data_ingestion = DataIngestion(config = data_ingestion_config) # from componennts/data_ingestion.py
            data_ingestion.download_file() # from componennts/data_ingestion.py
            


if __name__ == "__main__":
    try:
        logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.initiate_data_ingestion()
        logging.info(f">>>>>>>>>>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<<<<<\n\nx=====x")
    except Exception as e:
        logging.exception(e)
        raise e
