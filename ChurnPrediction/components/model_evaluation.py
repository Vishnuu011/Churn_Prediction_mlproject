import os
import sys
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
import numpy as np
import pickle
from mlflow.models.signature import infer_signature
import dagshub


from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

from urllib.parse import urlparse
from ChurnPrediction.constants import *
from ChurnPrediction.exception import CustomException
from ChurnPrediction.logger import logging
from ChurnPrediction.entity.config_entity import DataTansformationConfig
from ChurnPrediction.entity.artifact_entity import DataTransformationArifact
from ChurnPrediction.utils.utils import load_numpy_array

class ModelEvaluation:

    def __init__(self):
        pass

    @staticmethod
    def evaluate_clf(actual, predicted):

        try:
            acc = accuracy_score(actual, predicted)
            f1 = f1_score(actual, predicted)
            precision = precision_score(actual, predicted)
            recall = recall_score(actual, predicted)
            roc_auc = roc_auc_score(actual, predicted)
            return acc, f1 , precision, recall, roc_auc
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_evaluation(self,train_arr, test_arr):

        try:
            train_arr = load_numpy_array(file_path=train_arr)
            test_arr = load_numpy_array(file_path=test_arr)

            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            with open("save_objects_arifact\model.pkl", "rb") as file:
                model = pickle.load(file)
            

            mlflow.set_registry_uri("https://dagshub.com/Vishnuu011/Churn_Prediction_mlproject.mlflow")
            dagshub.init(repo_owner='Vishnuu011', repo_name='Churn_Prediction_mlproject', mlflow=True)

            traking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
            print(traking_url_type)

            with mlflow.start_run():
                predicted = model.predict(X_test)

                (acc, f1 , precision, recall, roc_auc) = ModelEvaluation.evaluate_clf(actual=y_test, predicted=predicted)

                mlflow.log_metric("acc", acc)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("roc_auc", roc_auc)

                signature = infer_signature(X_test, model.predict(X_test))

# Log the model with signature and input example
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model_artifact",
                    input_example=X_test,
                    signature=signature
                  )

        except Exception as e:
            raise CustomException(e, sys)    



  