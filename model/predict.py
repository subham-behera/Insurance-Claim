# model/predict.py

import numpy as np
from model.train import InsuranceModel

def load_model_and_predict(input_data):
    model = InsuranceModel(data_path='data/insurance.csv')
    model.load_data()
    model.train_and_evaluate()
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    input_data = (37, 1, 30.8, 2, 1, 0)
    prediction = load_model_and_predict(input_data)
    print("The person will get insurance money = ", prediction)
