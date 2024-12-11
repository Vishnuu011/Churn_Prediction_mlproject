import os
import sys

from ChurnPrediction.exception import CustomException
from ChurnPrediction.logger import logging

from ChurnPrediction.constants import *

from ChurnPrediction.components.data_ingestion import DataIngestion
from ChurnPrediction.components.data_tansformation import DataTransformation
from ChurnPrediction.components.model_trainer import ModelTrainer
from ChurnPrediction.components.model_evaluation import ModelEvaluation
from ChurnPrediction.entity.config_entity import DataIngestionConfig, DataTansformationConfig, ModelTrainerConfig
from ChurnPrediction.entity.artifact_entity import DataIngestionArtifact, DataTransformationArifact, ModelTrainerArtifact




class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_tansformation_config = DataTansformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

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
        
    def start_data_tansformation(self, data_ingestion_artifact:DataIngestionArtifact) -> DataTransformationArifact:
        try:
            data_transformation = DataTransformation(data_ingestion_arifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_tansformation_config)
            data_transformation_arifact = data_transformation.initiate_data_transformation()
            return data_transformation_arifact
        except Exception as e:
            raise CustomException(e, sys)  


    def start_model_trainer(self, data_transformation_artifact : DataTransformationArifact) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(data_transformation_arifact=data_transformation_artifact,
                                         model_trainer_config=self.model_trainer_config
                                         )
            model_artifact = model_trainer.initiate_model_trainer()
            return model_artifact
        except Exception as e:
            raise CustomException(e, sys)

    
    def run_pipeline(self, ) -> None:
        
       try:
          data_ingestion_artifact = self.start_data_ingestion()
          data_transformation_artifact = self.start_data_tansformation(data_ingestion_artifact=data_ingestion_artifact)
          model_trainer = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
       except Exception as e:
           raise CustomException(e, sys)    