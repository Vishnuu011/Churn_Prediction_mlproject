import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN, SMOTETomek

from ChurnPrediction.constants import *
from ChurnPrediction.logger import logging
from ChurnPrediction.exception import CustomException
from ChurnPrediction.entity.config_entity import DataTansformationConfig
from ChurnPrediction.entity.artifact_entity import DataIngestionArtifact, DataTransformationArifact
from ChurnPrediction.entity.estimator import TargetValueMapping
from ChurnPrediction.utils.utils import read_yaml, save_numpy_array, save_object


import warnings
warnings.filterwarnings("ignore")

@dataclass
class DatatransformationObj:
    preprocessor_obj_file_path=os.path.join('save_objects_artifact',"preprocesser.pkl")

@dataclass 
class Datatransformationdataobj:
    data_obj = os.path.join("npydata", TRAIN_FILE.replace("csv", "npy")) 
    test_obj = os.path.join("npydata", TEST_FILE.replace("csv", "npy"))    


class DataTransformation:

    def __init__(self, data_ingestion_arifact :DataIngestionArtifact , 
                 data_transformation_config : DataTansformationConfig):
        
        try:
            self.data_ingestion_arifact = data_ingestion_arifact
            self.data_tansformation_config = data_transformation_config
            self._schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)
            self.data_transform_obj = DatatransformationObj()
            self.npy_data_obj = Datatransformationdataobj()
        except Exception as e:
            raise CustomException(e, sys)
        
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys) 
        

    def get_data_trasform_object(self)->Pipeline:

        logging.info(
            "Entering get_data_trasform_object operation ..."
        )

        try:
           or_columns = self._schema_config['or_columns']
           num_features = self._schema_config['num_features']

           num_pipline = Pipeline([
               ('imputer', SimpleImputer(strategy = 'median')),
               ('scaler', StandardScaler())
            ])

           cat_pipline = Pipeline([
               ('encoder', OrdinalEncoder())
            ])

           preprosser = ColumnTransformer([
                ('numeric', num_pipline, num_features),
                ('categorical', cat_pipline, or_columns)
            ])
           logging.info("Created preprocessor object from ColumnTransformer")
           logging.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
           return preprosser
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self,)->DataTransformationArifact:

        try:
            logging.info("Entering initiate_data_transformation operation ...")

            train_df = DataTransformation.read_data(file_path=self.data_ingestion_arifact.train_file_path)
            test_df = DataTransformation.read_data(file_path=self.data_ingestion_arifact.test_file_path)

            preprosser = self.get_data_trasform_object()

            input_features_train_df = train_df.drop(columns=[TRAGET_COL], axis=1)
            input_target_features_train_df = train_df[TRAGET_COL]

            input_target_features_train_df_l = input_target_features_train_df.replace(
                TargetValueMapping()._asdict()
            )

            logging.info("Got train features and test features of Training dataset")
            
            input_features_test_df = test_df.drop(columns=[TRAGET_COL], axis=1)
            input_target_features_test_df = test_df[TRAGET_COL]

            input_target_features_test_df_l = input_target_features_test_df.replace(
                TargetValueMapping()._asdict()
            )

            logging.info(
                  "Applying preprocessing object on training dataframe and testing dataframe"
                ) 
            
            input_feature_train_arr = preprosser.fit_transform(input_features_train_df)

            logging.info(
                  "Applying preprocessing object on training dataframe and testing dataframe"
                ) 
            
            input_feature_test_arr = preprosser.transform(input_features_test_df)

            logging.info("Used the preprocessor object to transform the test features")

            logging.info("Applying SMOTEENN on Training dataset")

            smt = SMOTEENN(sampling_strategy="minority")

            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, input_target_features_train_df_l
            )

            logging.info("Applied SMOTEENN on training dataset")

            logging.info("Applying SMOTEENN on testing dataset")

            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                    input_feature_test_arr,  input_target_features_test_df_l 
                )
            
            logging.info("Applied SMOTEENN on testing dataset")

            logging.info("Created train array and test array")

            train_arr = np.c_[
                input_feature_train_final, np.array(target_feature_train_final)
            ]

            test_arr = np.c_[
                input_feature_test_final, np.array(target_feature_test_final)
            ]

            save_object(
                self.data_tansformation_config.transformed_object_file_path,
                preprosser
            )
            
            save_object(
                self.data_transform_obj.preprocessor_obj_file_path,
                preprosser
            )

            save_numpy_array(
                self.data_tansformation_config.data_transform_train_file_path,
                train_arr
            )

            save_numpy_array(
                self.data_tansformation_config.data_transform_test_file_path,
                test_arr
            )

            save_numpy_array(
                self.npy_data_obj.data_obj,
                train_arr
            )

            save_numpy_array(
                self.npy_data_obj.test_obj,
                test_arr
            )


            logging.info("Saved the preprocessor object")

            logging.info(
                "Exited initiate_data_transformation method of Data_Transformation class"
            )

            data_transformation_artifact = DataTransformationArifact(
                transformed_obj_file_path=self.data_tansformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_tansformation_config.data_transform_train_file_path,
                transformed_test_file_path=self.data_tansformation_config.data_transform_test_file_path
            )
            return data_transformation_artifact
        
        except Exception as e:
            raise CustomException(e, sys)        
        



