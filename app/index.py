import streamlit as st
import pickle
import os

# Load the model
model_path = 'app/model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
else:
    st.error(f"Model file '{model_path}' not found.")
    st.stop()

st.title('Medical Insurance Cost Prediction')

# Collect user inputs
age = st.number_input('Age', min_value=1, max_value=100, value=25)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Smoker', ['yes', 'no'])
region = st.selectbox('Region', ['southeast', 'southwest', 'northeast', 'northwest'])

# Mapping user inputs to numerical values
sex = 0 if sex == 'male' else 1
smoker = 0 if smoker == 'yes' else 1
region_map = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
region = region_map[region]

# Create input data tuple
input_data = (age, sex, bmi, children, smoker, region)

# Button for making the prediction
if st.button('Predict'):
    prediction = model.predict([input_data])
    st.write(f'The estimated insurance cost is {prediction[0]:.2f}')
