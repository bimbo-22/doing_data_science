import os,sys
import yaml
import json
import joblib
from box import ConfigBox
from typing import Any
from creditfraud.logging.logger import logging as logger
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from pathlib import Path
from creditfraud.exception.exception import CreditFraudException
import numpy as np
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CreditFraudException(e, sys)
    
def write_yaml_file(file_path: str, content:object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CreditFraudException(e, sys)
    
# for data transformation test and train
def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CreditFraudException(e, sys) from e
    
def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CreditFraudException(e, sys) from e 
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File {file_path} does not exist.")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CreditFraudException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CreditFraudException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            model_params = params[model_name]
            
            gs = GridSearchCV(model, model_params, cv=3, verbose=0)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            
            y_test_pred = model.predict(X_test)
            
            train_model_score = f1_score(y_train, y_train_pred, average='weighted')
            
            test_model_score = f1_score(y_test, y_test_pred, average='weighted')
            
            report[model_name] = test_model_score
        return report
            
    except Exception as e:
        raise CreditFraudException(e, sys) from e