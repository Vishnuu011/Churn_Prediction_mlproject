import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from tqdm import tqdm


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from ChurnPrediction.exception import CustomException
from ChurnPrediction.logger import logging
from ChurnPrediction.entity.config_entity import ModelTrainerConfig
from ChurnPrediction.entity.artifact_entity import DataTransformationArifact, ModelTrainerArtifact
from ChurnPrediction.constants import *
from ChurnPrediction.utils.utils import save_object, load_numpy_array, evaluate_models

import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelTrainerObj:
    trained_model_file_path=os.path.join("save_objects_arifact","model.pkl")


class ModelTrainer:

    def __init__(self, data_transformation_arifact:DataTransformationArifact, 
                 model_trainer_config:ModelTrainerConfig):

        try:
            self.data_transformation_arifact = data_transformation_arifact
            self.model_trainer_config = model_trainer_config
            self.model_obj_save = ModelTrainerObj()
        except Exception as e:
            raise CustomException(e, sys)  


    def get_model_object_and_report(self, train_array: np.array, test_array: np.array)->Tuple[object, object]:

        try:
            logging.info("spliting data train and test")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            param_grids = {
                'LogisticRegression': {
                    "class_weight":["balanced"],
                    'penalty': ['l1', 'l2'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga']
                },
                'KNeighborsClassifier': {},
                'SVC': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                },
                'RandomForestClassifier': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'GradientBoostingClassifier': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5],
                    'min_samples_split': [2]
                },
                'AdaBoostClassifier': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },
                'DecisionTreeClassifier': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2]
                },
                'GaussianNB': {},
                'XGBClassifier': {
                   'n_estimators': [100, 200],
                   'learning_rate': [0.01, 0.1],
                   'max_depth': [3, 6]
                },
                'LGBMClassifier': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'num_leaves': [31, 63]
                }
            }

            models = {
                'LogisticRegression': LogisticRegression(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'SVC': SVC(),
                'RandomForestClassifier': RandomForestClassifier(),
                'GradientBoostingClassifier': GradientBoostingClassifier(),
                'AdaBoostClassifier': AdaBoostClassifier(),
                'XGBClassifier': XGBClassifier(),
                'LGBMClassifier': LGBMClassifier(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'GaussianNB': GaussianNB()
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=param_grids)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            print(model_report)

            return best_model, best_model_score

        except Exception as e:
            raise CustomException(e, sys)  


    def initiate_model_trainer(self,)->ModelTrainerArtifact:

        try:
            
            train_arr = load_numpy_array(file_path=self.data_transformation_arifact.transformed_train_file_path)
            test_arr = load_numpy_array(file_path=self.data_transformation_arifact.transformed_test_file_path)

            best_model, best_model_score = self.get_model_object_and_report(train_array=train_arr, test_array=test_arr)

            if best_model_score<0.92:
                raise ValueError("No best model found")
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                self.model_trainer_config.trained_model,
                best_model
            )
            
            save_object(
                self.model_obj_save.trained_model_file_path,
                best_model
            )
            model_trainer_arifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model
            )
            return model_trainer_arifact

        except Exception as e:
            raise CustomException(e, sys)                 
