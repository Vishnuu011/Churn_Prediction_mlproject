from ChurnPrediction.pipline.training_pipeline import TrainPipeline
from ChurnPrediction.components.model_evaluation import ModelEvaluation
from ChurnPrediction.entity.artifact_entity import DataTransformationArifact
import os
#obj= TrainPipeline()
#obj.run_pipeline()
train = os.path.join("npydata", "train.npy")
test = os.path.join("npydata", "test.npy")
eval = ModelEvaluation()
eval.initiate_model_evaluation(train_arr=train, test_arr=test)