# Medical Insurance Cost Prediction

This project predicts medical insurance costs based on user inputs such as age, sex, BMI, number of children, smoking status, and region. The application is built with Python, utilizing Pandas and Scikit-learn for data processing and model training. The user interface is implemented with Streamlit, and the entire application is containerized using Docker. Continuous Integration and Continuous Deployment (CI/CD) are handled by GitHub Actions.

## Directory Structure

```
medical-insurance-cost-prediction/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── data/
│   └── insurance.csv
├── model/
│   ├── __init__.py
│   ├── train.py
│   └── predict.py
├── app/
│   └── index.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Project Setup

### Prerequisites

- Python 3.8+
- Docker

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Train the model and save it as `model.pkl`:

```bash
python model/train.py
```

### Running the Application

#### Locally

1. Run the Streamlit application:

```bash
streamlit run app/index.py
```

2. Access the application at `http://localhost:8501`.

#### Using Docker

1. Build the Docker image:

```bash
docker build -t medical-insurance-cost-prediction .
```

2. Run the Docker container:

```bash
docker run -p 8501:8501 medical-insurance-cost-prediction
```

3. Access the application at `http://localhost:8501`.

### CI/CD with GitHub Actions

This project uses GitHub Actions for continuous integration and continuous deployment. The workflow is defined in `.github/workflows/ci-cd.yml`.

To set up GitHub Actions:

1. Add the following secrets to your GitHub repository:
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`
   - `SERVER_IP`
   - `SERVER_USER`
   - `SERVER_PASSWORD`

2. On every push to the `main` branch, GitHub Actions will build the Docker image, push it to Docker Hub, and deploy it to your server.
