import pandas as pd
import os 
from creditfraud.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
from creditfraud.logging.logger import logging 

class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config
        
    def train_test_split(self):
        data = pd.read_csv(self.config.data_path)   
        
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        
        train.to_csv(os.path.join(self.config.root_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'), index=False)
        logger.info("Train-test split completed successfully.")
        logger.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
        
        print(train.shape, test.shape)
        
        return train, test