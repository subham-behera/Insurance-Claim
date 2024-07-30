import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class InsuranceModel:
    def __init__(self, data_path, model_dir='app'):
        self.data_path = data_path
        self.df = None
        self.models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "Support Vector Regressor": SVR()
        }
        self.best_model = None
        self.model_dir = model_dir

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self.df

    def preprocess_data(self):
        self.df['sex'] = self.df['sex'].map({'male': 0, 'female': 1})
        self.df['smoker'] = self.df['smoker'].map({'yes': 0, 'no': 1})
        self.df['region'] = self.df['region'].map({'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3})
        X = self.df.drop(columns='charges', axis=1)
        Y = self.df['charges']
        return X, Y

    def split_data(self, X, Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
        return X_train, X_test, Y_train, Y_test

    def evaluate_model(self, model, X_train, X_test, Y_train, Y_test):
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(Y_test, predictions)
        mse = mean_squared_error(Y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, predictions)
        return mae, mse, rmse, r2

    def train_and_evaluate(self):
        X, Y = self.preprocess_data()
        X_train, X_test, Y_train, Y_test = self.split_data(X, Y)
        results = {}
        best_r2 = -1

        for name, model in self.models.items():
            mae, mse, rmse, r2 = self.evaluate_model(model, X_train, X_test, Y_train, Y_test)
            results[name] = {
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R²": r2
            }
            if r2 > best_r2:
                best_r2 = r2
                self.best_model = model

        results_df = pd.DataFrame(results).T
        self.save_best_model()  # Save the best model
        return results_df

    def predict(self, input_data):
        input_data_as_array = np.asarray(input_data).reshape(1, -1)
        prediction = self.best_model.predict(input_data_as_array)
        return prediction[0]

    def save_best_model(self):
        if self.best_model:
            os.makedirs(self.model_dir, exist_ok=True)
            model_path = os.path.join(self.model_dir, 'model.pkl')
            with open(model_path, 'wb') as model_file:
                pickle.dump(self.best_model, model_file)
            print(f"Best model saved as '{model_path}'")
        else:
            print("No model has been trained yet.")

if __name__ == "__main__":
    model = InsuranceModel(data_path='../data/insurance.csv', model_dir='../app')
    model.load_data()
    results_df = model.train_and_evaluate()
    print(results_df)
