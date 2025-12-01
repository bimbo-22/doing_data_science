import os
import yaml
import json
import joblib
from box import ConfigBox
from typing import Any
from creditfraud.logging.logger import logging as logger
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from pathlib import Path


@ensure_annotations
def read_yaml(path_to_yaml: str) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.
    
    Args:
        path_to_yaml (str): Path to the YAML file.
        
    Returns:
        ConfigBox: Content of the YAML file as a ConfigBox object.
        
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully from {path_to_yaml}.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates directories if they do not exist.
    
    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): ignore if multiple dirs is to be created. 
        
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Creating directory: {path} for the file: {path}")
            
@ensure_annotations	
def save_json(path: Path, data: dict) :
    """
    Saves a dictionary as a JSON file.
    
    Args:
        path (Path): Path to save the JSON file.
        data (dict): Dictionary to save.
        
    """
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)
        logger.info(f"JSON file saved successfully at {path}.")
        
        
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file .
    
    Args:
        path (Path): Path to the JSON file.
        
    Returns:
        ConfigBox: data as class attributes instead of dict.
        
    """
    with open(path, "r") as json_file:
        content = json.load(json_file)
        logger.info(f"JSON file loaded successfully from {path}.")
        return ConfigBox(content)
    

@ensure_annotations
def save_bin(path: Path, data: Any):
    """
    Saves data as a binary file using joblib.
    
    Args:
        path (Path): Path to save the binary file.
        data (Any): Data to save.
        
    """
    joblib.dump(value = data, filename=path)
    logger.info(f"Binary file saved successfully at {path}.")
    
@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads data from a binary file using joblib.
    
    Args:
        path (Path): Path to the binary file.
        
    Returns:
        Any: Loaded data.
        
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded successfully from {path}.")
    return data