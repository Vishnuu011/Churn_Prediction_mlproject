import os
import sys

DATA_URL = "https://raw.githubusercontent.com/Vishnuu011/datastore/refs/heads/main/Telco_Customer_Churn.csv"

ARTIFACT_DIR : str = "artifact"

PIPELINE_NAME : str = "Churnpredictionpipeline"
MODEL_NAME_FILE = "model.pkl"
FILE_NAME = "churn.csv"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv" 
SCHEMA_FILE_PATH = os.path.join("config","schema.yaml")
PREPROSSER_OBJ_FILE_NAME = "preprocesser.pkl"
TRAGET_COL= "Churn"

# DATA INGESTION RELATED CONSTANTS

DATA_INGESTION_DIR_NAME: str ="data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME: str ="feature_store"
DATA_INGESTION_INGESTED_DIR_NAME: str ="ingested"
DATA_INGESTION_TRAIN_DIR_NAME: str ="train"
DATA_INGESTION_TEST_DIR_NAME: str ="test"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float =0.2

# DATA TRANSFORMATION RELATED CONSTANTS

DATA_TRANSFORMATION_DIR_NAME : str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR_NAME : str = "transform"
DATA_TRANSFORMATION_OBJECT_DIR_NAME : str = "transformed_obj"

# MODEL TRAINER RELATED CONSTANTS

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"