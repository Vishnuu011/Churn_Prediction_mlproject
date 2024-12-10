from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    train_file_path:str 
    test_file_path:str


@dataclass
class DataTransformationArifact:
    transformed_obj_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str     


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path:str 
        