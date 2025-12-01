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
    