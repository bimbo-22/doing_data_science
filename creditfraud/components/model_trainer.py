import os,sys
from creditfraud.logging.logger import logging
from creditfraud.exception.exception import CreditFraudException
from creditfraud.entity.config_entity import ModelTrainerConfig
from creditfraud.config.configuration import ConfigurationManager
from networksecurity.utils.main_utils.utils import load_object, save_object, load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay,precision_score, recall_score, f1_score, roc_auc_score,roc_curve 
import mlflow
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CreditFraudException(e, sys)
        
    def track_mlflow(self,best_model, classificationmetric):
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        with mlflow.start_run(run_name="Model_Training_1"):
            f1_score = classificationmetric.f1_score
            recall_score = classificationmetric.recall_score
            precision_score = classificationmetric.precision_score
            
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision_score)
            mlflow.log_metric("recall_score", recall_score)
            
            mlflow.sklearn.log_model(best_model, "model")
    def train_model(self, X_train, y_train, X_test, y_test):
        models={
            "LogisticRegression":LogisticRegression(),
            "DecisionTreeClassifier":DecisionTreeClassifier(),
            "RandomForestClassifier":RandomForestClassifier(),
            "GradientBoostingClassifier":GradientBoostingClassifier(),
            "AdaBoostClassifier":AdaBoostClassifier(),
            "Xgboost":XGBClassifier()
        }
        params = {
            "DecisionTreeClassifier": 
            {
            'criterion':['gini','entropy', 'log_loss'],
            'splitter':['best','random'],
            'max_depth':[1,2,3,4,5],
            'max_features':['auto','sqrt','log2']
            },
                
            "RandomForestClassifier": 
            {"max_depth": [5, 8, 15, None, 10],
             "max_features": [5, 7, 8],
             "min_samples_split": [2, 8, 15, 20],
             "n_estimators": [100, 200, 500, 1000]
             },
                
            "GradientBoostingClassifier": {
                "loss": ['log_loss','deviance','exponential'],
                    "criterion": ['friedman_mse','squared_error','mse'],
                    "min_samples_split": [2, 8, 15, 20],
                    "n_estimators": [100, 200, 500],
                    "max_depth": [5, 8, 15, None, 10]
                        },
                "LogisticRegression": {
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'C': [100,10,1.0,0.1,0.01],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                },
                    
                "AdaBoostClassifier": {
                "n_estimators":[50,60,70,80,90],
                "algorithm":['SAMME']
                },
                "Xgboost":{"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20, 30],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]}
                  }
        model_report:dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
        
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]
        
        y_train_pred = best_model.predict(X_train)
        
        classification_train_metric = get_classification_score(y_true = y_train, y_pred = y_train_pred)
        
        # function to track mlflow
        self.track_mlflow(best_model, classification_train_metric)


        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true = y_test, y_pred = y_test_pred)
        
        self.track_mlflow(best_model, classification_test_metric)
        
        # load the pickle preprocessor object so we can use for new data coming in 
        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        
        model_dir_path =  os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)
        
        # call network model to apply same feature engineering using the preprocessor 
        Network_Model = NetworkModel(preprocessor = preprocessor, model = best_model)
        
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_Model)
        
        #model pusher or push to aws boto3 
        save_object("final_model/model.pkl", best_model)
        
        # Model Trainer Artifact 
        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, train_metric_artifact=classification_train_metric, test_metric_artifact=classification_test_metric)
        
        logging.info(f"Model Trainer artifact: {model_trainer_artifact}")
        
        return model_trainer_artifact
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            # load training and testing numpy arrays
            train_array = load_numpy_array_data(train_file_path)
            test_array = load_numpy_array_data(test_file_path)
            
            x_train,y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            model = self.train_model(X_train=x_train, y_train = y_train,X_test = x_test, y_test = y_test)
            
            
        except Exception as e:
            raise CreditFraudException(e, sys)