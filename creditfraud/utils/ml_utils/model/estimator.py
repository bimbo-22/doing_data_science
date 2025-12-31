from creditfraud.logging.logger import logging
from creditfraud.exception.exception import CreditFraudException
from creditfraud.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
import os, sys

class FraudModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CreditFraudException(e, sys)
    def predict(self, X):
        try:
            X_transformed = self.preprocessor.transform(X)
            y_hat = self.model.predict(X_transformed)
            return y_hat
        except Exception as e:
            raise CreditFraudException(e, sys)