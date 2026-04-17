# MLOps Pipeline — Industrialization of a Machine Learning Model

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.x-orange?style=flat-square&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-Makefile-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Modularization](#modularization)
5. [CI/CD Automation](#cicd-automation)
6. [Experiment Tracking](#experiment-tracking)
7. [API Deployment](#api-deployment)
8. [Docker Containerization](#docker-containerization)
9. [Monitoring Stack](#monitoring-stack)
10. [Technologies](#technologies)
11. [Getting Started](#getting-started)

---

## Overview

This project transforms an existing Machine Learning notebook into a fully industrialized **MLOps pipeline**. The goal is to apply software engineering best practices to a data science workflow, producing a system that is modular, reproducible, testable, and ready for production deployment.

The pipeline covers the complete lifecycle of a machine learning model:

| Phase | Description |
|---|---|
| **Modularization** | Refactoring notebook code into independent, reusable Python modules |
| **Code Quality** | Automated linting, formatting, and type checking via Makefile |
| **Experiment Tracking** | Logging of metrics, parameters, and model artifacts with MLflow |
| **API Serving** | REST API built with FastAPI for real-time inference |
| **Containerization** | Docker image for reproducible, portable deployment |
| **Monitoring** | Performance tracking and log analysis with Elasticsearch and Kibana |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MLOps Pipeline                           │
│                                                                 │
│  ┌───────────────┐    ┌───────────────┐    ┌────────────────┐  │
│  │  Data Input   │───▶│   Training    │───▶│  MLflow Store  │  │
│  └───────────────┘    │   Pipeline    │    │ (metrics,      │  │
│                       └───────┬───────┘    │  models,       │  │
│                               │            │  artifacts)    │  │
│                       ┌───────▼───────┐    └────────────────┘  │
│                       │  Model        │                         │
│                       │  Registry     │                         │
│                       └───────┬───────┘                         │
│                               │                                 │
│                       ┌───────▼───────┐    ┌────────────────┐  │
│                       │  FastAPI      │───▶│  Monitoring    │  │
│                       │  REST API     │    │  Elasticsearch │  │
│                       │  /predict     │    │  + Kibana      │  │
│                       └───────────────┘    └────────────────┘  │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  CI/CD — Makefile (Pylint · Flake8 · MyPy · Black)        │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Modularization

The original Jupyter notebook was decomposed into independent, single-responsibility Python modules to improve maintainability, testability, and reusability.

| Module | Responsibility |
|---|---|
| `model_pipeline.py` | Data preprocessing, model training, and inference |
| `retrain_model.py` | Automated retraining logic triggered by drift or schedule |
| `app.py` | Central orchestrator — coordinates all pipeline stages |
| `test_pipeline.py` | Unit tests for pipeline components |

Each module is designed to be imported independently, enabling isolated testing and modular reuse across different pipeline configurations.

---

## CI/CD Automation

Code quality is enforced through a `Makefile` that consolidates all quality checks into a single command. This ensures a consistent standard across development and can be integrated into any CI system (GitHub Actions, GitLab CI, etc.).

### Available Targets

| Target | Tool | Purpose |
|---|---|---|
| `lint` | **Pylint** | Static code analysis and error detection |
| `style` | **Flake8** | PEP 8 style compliance |
| `type` | **MyPy** | Static type checking |
| `format` | **Black** | Automatic code formatting |
| `test` | **Pytest** | Run the test suite |
| `all` | All of the above | Full quality gate |

### Usage

```bash
# Run the full quality pipeline
make all

# Run individual targets
make lint
make format
make test
```

---

## Experiment Tracking

All training runs are tracked using **MLflow**, providing a centralized registry for metrics, hyperparameters, model versions, and artifacts.

### What Is Tracked

- **Metrics** — accuracy, precision, recall, F1-score, loss curves
- **Parameters** — hyperparameter values used during training
- **Artifacts** — saved models, plots, evaluation reports
- **Model versions** — promotion across Staging, Production, and Archived stages

### Launch the MLflow UI

```bash
mlflow ui
```

Then navigate to [http://localhost:5000](http://localhost:5000) to view the experiment dashboard.

### Example — Logging a Run

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.sklearn.log_model(model, "model")
```

---

## API Deployment

The trained model is served through a production-ready **REST API** built with FastAPI, enabling real-time inference over HTTP.

### Start the API

```bash
uvicorn api.main:app --reload
```

The API will be available at [http://localhost:8000](http://localhost:8000).

Interactive documentation (Swagger UI) is automatically available at:

```
http://localhost:8000/docs
```

### Endpoint Reference

#### `POST /predict`

Accepts a JSON payload with input features and returns the model prediction.

**Request body:**

```json
{
  "feature_1": 5.1,
  "feature_2": 3.5,
  "feature_3": 1.4
}
```

**Response:**

```json
{
  "prediction": 1,
  "confidence": 0.94,
  "model_version": "v2.1"
}
```

---

## Docker Containerization

The application is fully containerized using Docker, ensuring a consistent and reproducible runtime environment across development, staging, and production.

### Build the Image

```bash
docker build -t mlops-project .
```

### Run the Container

```bash
docker run -p 8000:8000 mlops-project
```

### Start the Full Stack (API + Monitoring)

```bash
docker-compose up --build
```

### Publish to DockerHub

```bash
docker tag mlops-project <your-dockerhub-username>/mlops-project:latest
docker push <your-dockerhub-username>/mlops-project:latest
```

---

## Monitoring Stack

Production monitoring is implemented using a combination of MLflow (model-level) and the **ELK Stack** (system-level), providing end-to-end observability.

### Components

| Component | Role |
|---|---|
| **MLflow** | Tracks model metrics, detects performance degradation, and manages version comparison |
| **Elasticsearch** | Stores and indexes prediction logs, request data, and system events |
| **Kibana** | Provides real-time dashboards for metrics visualization and alerting |

### Monitored Signals

- Model prediction accuracy over time
- Input data distribution shift (feature drift)
- API request volume and response latency
- Error rates and exception logs
- Model version performance comparison

### Access Kibana Dashboard

```bash
# After running docker-compose up
http://localhost:5601
```

---

## Technologies

| Category | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **ML Framework** | Scikit-learn |
| **Experiment Tracking** | MLflow |
| **API Framework** | FastAPI + Uvicorn |
| **Containerization** | Docker, DockerHub |
| **Orchestration** | Docker Compose |
| **CI/CD** | Makefile (Pylint, Flake8, MyPy, Black) |
| **Monitoring** | Elasticsearch, Kibana |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- `make` (available by default on Linux/macOS; install via [GnuWin32](http://gnuwin32.sourceforge.net/packages/make.htm) on Windows)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/mlops-pipeline.git
cd mlops-pipeline

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Run code quality checks
make all

# Train the model
python app.py

# Launch MLflow UI
mlflow ui

# Start the API
uvicorn api.main:app --reload

# Start the full monitoring stack
docker-compose up --build
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
