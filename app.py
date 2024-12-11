import streamlit as st

import pandas as pd



from ChurnPrediction.exception import CustomException
import sys
from ChurnPrediction.pipline.prediction_pipeline import predictPipline



class CustomData:
    def __init__(  self,
        gender: str,
        InternetService: str,
        Contract:str,
        tenure: int,
        MonthlyCharges: float,
        TotalCharges: float):

        self.gender = gender

        self.InternetService = InternetService

        self.Contract = Contract

        self.tenure = tenure

        self.MonthlyCharges = MonthlyCharges

        self.TotalCharges = TotalCharges

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "InternetService": [self.InternetService],
                "Contract": [self.Contract],
                "tenure": [self.tenure],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }

            return pd.DataFrame(custom_data_input_dict, index=[0])

        except Exception as e:
            raise CustomException(e, sys)
        

st.title("Customer Churn Prediction App")

st.header("Enter Customer Information")
with st.form(key="my_form"):
    data=CustomData(
        gender=st.selectbox("Contract", ('Male', 'Female')),
        InternetService=st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No')),
        Contract=st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year')),
        tenure=st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1),
        MonthlyCharges=st.number_input("Monthly Charges", min_value=0, max_value=200, value=50),
        TotalCharges=st.number_input("Total Charges", min_value=0, max_value=10000, value=0)
    ) 
    submitted = st.form_submit_button("Submit")

data = data.get_data_as_data_frame()
data_reshaped = data.values.reshape(1, -1)
prediction = predictPipline()
prediction = prediction.predict(data)

st.header("Prediction Result")
if prediction[0] == 0:
    st.success("**This customer is likely to stay.**")
else:
    st.error("**This customer is likely to churn.**")


    