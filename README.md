# 🚀 MLOps Pipeline – Industrialization of a Machine Learning Model

This project transforms an existing Machine Learning notebook into a fully industrialized **MLOps pipeline**, including modularization, CI/CD automation, experiment tracking, API deployment, containerization, and monitoring.

---

## 📌 Project Overview

The goal of this project is to refactor my original ML project into a production-ready system with:

- **Clean modular code** (training, inference, retraining)
- **Automated CI/CD** using Makefile (linting, type checks, formatting)
- **Model tracking and versioning** with MLflow
- **REST API deployment** using FastAPI
- **Docker containerization** and DockerHub publishing
- **Monitoring** with MLflow, Elasticsearch, and Kibana

---


## 🔧 Modularization

The initial Jupyter notebook was refactored into independent, reusable Python modules:

- `model_pipeline.py` – preprocessing, model training & inference
- `retrain_model.py` – automated retraining logic
- `app.py` – central pipeline orchestrator
- `test_pipeline.py` – testing functions


---

## 🔄 CI/CD Automation (Makefile)

The Makefile ensures code quality with:

- **Pylint** – linting  
- **Flake8** – style checking  
- **MyPy** – type checking  
- **Black** – code formatting  

Run everything with:

make all


---

## 📊 Experiment Tracking with MLflow

MLflow tracks:

- Training metrics  
- Hyperparameters  
- Model versions  
- Artifacts (plots, reports)

Launch MLflow UI:

mlflow ui

---

## ⚡ API Deployment (FastAPI)

The model is served through a production-ready REST API.

Run the API locally:

uvicorn api.main:app --reload



Main endpoint:

POST /predict


---

## 🐳 Docker Containerization

Build the Docker image:

docker build -t mlops-project .



Run the container:

docker run -p 8000:8000 mlops-project


Push to DockerHub:

docker push <username>/mlops-project


---

## 📈 Monitoring (MLflow + Elasticsearch + Kibana)

The monitoring stack provides:

- Model performance tracking  
- Metrics visualization  
- Logs & drift analysis  
- Version comparison  

---

## 🧰 Technologies Used

- Python  
- FastAPI  
- MLflow  
- Scikit-learn  
- Docker  
- DockerHub  
- Makefile  
- Elasticsearch  
- Kibana  
