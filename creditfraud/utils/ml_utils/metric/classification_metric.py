from creditfraud.exception.exception import CreditFraudException
from creditfraud.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from creditfraud.logging.logger import logging
import sys

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        f1 = f1_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        roc_auc = roc_auc_score(y_true, y_pred)
        
        classification_metric_artifact = ClassificationMetricArtifact(
            f1_score=f1,
            accuracy_score=accuracy,
            precision_score=precision,
            recall_score=recall,
            roc_auc_score=roc_auc
        )
        print("Classification metrics calculated successfully")
        print(classification_metric_artifact)
        logging.info("Classification metrics calculated successfully")
        logging.info(classification_metric_artifact)
        return classification_metric_artifact
    except Exception as e:
        logging.error("Error in calculating classification metrics")
        raise CreditFraudException(e, sys)