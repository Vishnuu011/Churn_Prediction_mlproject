import os
import sys
import sklearn
import pickle

from ChurnPrediction.exception import CustomException
from ChurnPrediction.utils.utils import load_object


class predictPipline:

    def __init__(self,):

        pass
        
    def predict(self, feature):

        try:
            with open("save_objects_artifact\model.pkl", 'rb') as file:
              model = pickle.load(file)


            with open('save_objects_artifact\preprocesser.pkl', 'rb') as file:
              preposser =pickle.load(file)

            scaled_data = preposser.transform(feature)
            preds = model.predict(scaled_data)
            

            return preds

        except Exception as e:
            raise CustomException(e, sys)
        


