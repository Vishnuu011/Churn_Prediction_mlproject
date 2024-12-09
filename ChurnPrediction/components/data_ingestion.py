import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ChurnPrediction.constants import *
from ChurnPrediction.logger import logging
from ChurnPrediction.exception import CustomException
from ChurnPrediction.entity.config_entity import DataIngestionConfig
from ChurnPrediction.entity.artifact_entity import DataIngestionArtifact
from ChurnPrediction.utils.utils import read_yaml

import warnings
warnings.filterwarnings("ignore")

class DataIngestion:

    def __init__(self, data_ingestion_config : DataIngestionConfig=DataIngestionConfig()):

        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)
        

    def export_data_in_feature_store(self)->pd.DataFrame:

        try:
            logging.info("Entering export_data_in_feature_store operation ......")

            data = pd.read_csv(DATA_URL)
            
            data_selected = data[["customerID","gender","InternetService","Contract","tenure","MonthlyCharges","TotalCharges"]]
            logging.info(data_selected)
            logging.info(f"Check data frame shape {data_selected.shape}")
            data_selected["TotalCharges"] = pd.to_numeric(data_selected["TotalCharges"], errors="coerce")
            logging.info(data_selected.info())

            feature_store_file_data = self.data_ingestion_config.data_ingestion_feature_store
            dir_n = os.path.dirname(feature_store_file_data)
            os.makedirs(dir_n, exist_ok=True)
            logging.info(f"saving data into feature store {feature_store_file_data}")

            data_selected.to_csv(feature_store_file_data, index=False, header=True)
            return data_selected

        except Exception as e:
            raise CustomException(e, sys) 


    def split_train_and_test(self, dataframe : pd.DataFrame)->None:

        try:
            logging.info("Entering split_train_and_test operation .....")

            train, test = train_test_split(dataframe, test_size=0.2)
            
            logging.info("splited train and test data")

            logging.info(f"Checking both train and test data shape train:{train.shape}, test:{test.shape}")

            dir_t = os.path.dirname(self.data_ingestion_config.data_ingestion_train_file_path)
            os.makedirs(dir_t, exist_ok=True)

            train.to_csv(self.data_ingestion_config.data_ingestion_train_file_path, index=False, header=True)
            test.to_csv(self.data_ingestion_config.data_ingestion_test_file_path, index=False, header=True)
            logging.info(f"Exported train data and test data to path")
        except Exception as e:
            raise CustomException(e, sys) 


    def initiate_data_ingestion(self)->DataIngestionArtifact:

        try:
            logging.info("Entering initiate_data_ingestion operation")

            dataframe = self.export_data_in_feature_store()
            self.split_train_and_test(dataframe=dataframe)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.data_ingestion_train_file_path,
                test_file_path=self.data_ingestion_config.data_ingestion_test_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

