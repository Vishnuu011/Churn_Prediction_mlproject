import os
import sys

from ChurnPrediction.exception import CustomException
from ChurnPrediction.logger import logging

from ChurnPrediction.constants import *

from ChurnPrediction.components.data_ingestion import DataIngestion
from ChurnPrediction.entity.config_entity import DataIngestionConfig
from ChurnPrediction.entity.artifact_entity import DataIngestionArtifact




class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        

    def start_data_ingestion(self) -> DataIngestionArtifact:
        
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e 
        

    def run_pipeline(self, ) -> None:
        
       try:
          data_ingestion_artifact = self.start_data_ingestion()
       except Exception as e:
           raise CustomException(e, sys)    