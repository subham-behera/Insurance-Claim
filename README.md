# Medical Insurance Cost Prediction

This project predicts the cost of medical insurance for individuals based on various features such as age, sex, BMI, number of children, smoking status, and region. The project includes a machine learning model trained on an insurance dataset and a Streamlit application for user interaction.

## Project Structure

```
insurance-cost/
├── data/
│   └── insurance.csv
├── model/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
├── app/
│   └── index.py
├── requirements.txt
└── README.md
```
- **data/**: Directory containing the insurance dataset (`insurance.csv`).
- **model/**: Directory containing the model training and prediction scripts.
  - `train.py`: Script to train the model.
  - `predict.py`: Script to make predictions using the trained model.
- **app/**: Directory containing the Streamlit application.
  - `index.py`: Main script for the Streamlit application.
- **requirements.txt**: List of Python dependencies.
- **README.md**: Project documentation (this file).

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Streamlit
- Scikit-learn
- Pandas
- Numpy

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/subham-behera/Insurance-cost.git
   cd Insurance-Cost
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (optional, if you want to retrain the model):
   ```bash
   python model/train.py
   ```

4. Run the Streamlit application:
   ```bash
   streamlit run app/index.py
   ```

### Usage

1. Open the Streamlit application in your web browser.
2. Input the required details:
   - Age
   - Sex
   - BMI
   - Number of Children
   - Smoking Status
   - Region
3. Click on the "Predict" button to get the estimated insurance cost.
