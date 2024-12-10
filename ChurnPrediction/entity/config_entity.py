import os
import sys
from dataclasses import dataclass
from ChurnPrediction.constants import *
from datetime import datetime


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    data_ingestion_feature_store = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR_NAME, FILE_NAME)
    data_ingestion_train_file_path = os.path.join(data_ingestion_dir, DATA_INGESTION_DIR_NAME, TRAIN_FILE)
    data_ingestion_test_file_path = os.path.join(data_ingestion_dir, DATA_INGESTION_DIR_NAME, TEST_FILE)
    train_test_split_ratio : float = 0.2


@dataclass
class DataTansformationConfig:
    data_transform_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    data_transform_train_file_path = os.path.join(data_transform_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME, 
                                                  TRAIN_FILE.replace("csv", "npy")) 
    data_transform_test_file_path = os.path.join(data_transform_dir,DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME, 
                                                 TEST_FILE.replace("csv", "npy"))
    transformed_object_file_path = os.path.join(data_transform_dir, DATA_TRANSFORMATION_OBJECT_DIR_NAME, PREPROSSER_OBJ_FILE_NAME)


