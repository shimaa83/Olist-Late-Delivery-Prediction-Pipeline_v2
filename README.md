#  Olist Late Delivery Prediction Pipeline

An end-to-end **MLOps pipeline** for predicting late delivery of Olist orders.

The project covers the complete machine learning lifecycle:

**Data → Validation → Feature Engineering → Training → MLflow → DVC → ONNX → FastAPI → Docker → Monitoring**

---

##  Project Overview

The objective of this project is to build a production-oriented machine learning service that predicts whether an Olist order is likely to be delivered late.

The trained model is exported to **ONNX** and served through a **FastAPI REST API**.

The service also includes:

* Data validation
* Feature engineering
* ML model training
* MLflow experiment tracking
* DVC data and artifact versioning
* ONNX model export
* FastAPI inference
* Docker containerization
* Prometheus metrics
* Grafana monitoring
* Prediction distribution tracking
* Prediction logging
* Drift monitoring strategy
* Health checks
* Automated testing
* Ruff code quality checks
* GitHub Actions CI/CD

---

#  Architecture

```text
                         Olist Dataset
                              │
                              ▼
                     Data Processing
                              │
                              ▼
                     Data Validation
                    Great Expectations
                              │
                              ▼
                     Feature Engineering
                              │
                              ▼
                       Model Training
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                 MLflow                DVC
            Experiment Tracking   Data & Artifacts
                    │
                    ▼
                  Best Model
                    │
                    ▼
                ONNX Export
                    │
                    ▼
              ┌──────────────┐
              │   FastAPI    │
              │     API      │
              └──────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       /predict   /metrics   /health
          │          │
          │          ▼
          │      Prometheus
          │          │
          │          ▼
          │       Grafana
          │
          ▼
   Prediction Logs
          │
          ▼
   Distribution / Drift
          │
          ▼
      Alerting
```

---

#  Technology Stack

| Technology         | Purpose                               |
| ------------------ | ------------------------------------- |
| Python 3.13+       | Programming language                  |
| uv                 | Dependency and environment management |
| FastAPI            | REST API                              |
| Pydantic           | API validation                        |
| Scikit-learn       | Machine learning                      |
| XGBoost            | Machine learning                      |
| ONNX               | Model serialization                   |
| ONNX Runtime       | Model inference                       |
| MLflow             | Experiment tracking                   |
| DVC                | Data/artifact versioning              |
| Great Expectations | Data validation                       |
| Prometheus         | Metrics collection                    |
| Grafana            | Monitoring dashboard                  |
| Docker             | Containerization                      |
| Docker Compose     | Multi-service orchestration           |
| pytest             | Testing                               |
| Ruff               | Linting and formatting                |
| pre-commit         | Automated code quality                |
| GitHub Actions     | CI/CD                                 |

---

#  Project Structure

```text
Olist-Late-Delivery-Prediction-Pipeline_v2/
│
├── app/
│   ├── main.py
│   └── schema.py
│
├── src/
│   └── task3/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── split_data.py
│       ├── train_mlflow.py
│       ├── validate_data.py
│       └── export_onnx.py
│
├── tests/
│   ├── test_data.py
│   ├── test_export_onnx.py
│   ├── test_features.py
│   ├── test_model_training.py
│   └── test_preprocessing.py
│
├── artifacts/
├── models/
├── data/
├── logs/
│
├── Dockerfile
├── docker-compose.yml
├── prometheus.yml
├── pyproject.toml
├── uv.lock
├── dvc.yaml
├── .pre-commit-config.yaml
└── README.md
```

---

#  Python & Dependency Management

The project uses **uv** for dependency management.

Python requirement:

```text
Python >= 3.13
```

Install the project dependencies:

```bash
uv sync
```

Run Python through the project environment:

```bash
uv run python
```

Run tests:

```bash
uv run pytest
```

---

#  Testing

The project includes automated tests covering:

* Data processing
* Feature engineering
* Preprocessing
* Model training
* ONNX export

Run:

```bash
uv run pytest
```

Current test result:

```text
13 passed in 4.99s
```

---

#  Code Quality

The project uses **Ruff** for linting and formatting.

Run Ruff:

```bash
uv run ruff check .
```

Automatically fix supported issues:

```bash
uv run ruff check . --fix
```

---

#  Pre-commit

The project uses `pre-commit` to automatically check the code before commits.

Configured hooks include:

* trailing whitespace
* end-of-file fixing
* YAML validation
* large file detection
* Ruff linting
* Ruff formatting

Install the hooks:

```bash
uv run pre-commit install
```

Run all hooks manually:

```bash
uv run pre-commit run --all-files
```

---

#  DVC

**DVC (Data Version Control)** is used to version datasets and machine learning artifacts.

This makes it possible to trace experiments and model results back to specific data versions.

Useful commands:

```bash
uv run dvc status
```

```bash
uv run dvc repro
```

```bash
uv run dvc push
```

---

#  MLflow

**MLflow** is used for experiment tracking.

The training pipeline can track:

* Parameters
* Metrics
* Model artifacts
* Training runs
* Experiment history

This makes model experiments reproducible and easier to compare.

---

# ✅ Data Validation

**Great Expectations** is used to validate incoming data before it reaches the model.

Validation can include:

* Column types
* Value ranges
* Allowed categories
* Missing values
* Data quality constraints

Invalid data should be rejected or flagged before inference.

---

#  Model & ONNX

The trained model is exported to **ONNX**.

The FastAPI service loads:

* ONNX model
* Categorical imputer
* Categorical encoder
* Numerical imputer
* Numerical scaler
* Feature metadata

The model is loaded when the API starts.

Current model version:

```text
v1.0.0-onnx
```

Inference is performed using:

```text
ONNX Runtime
```

---

#  FastAPI

The application provides a REST API using FastAPI.

Run locally:

```bash
uv run uvicorn app.main:app --reload --port 8000
```

API:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

#  API Endpoints

## Health

```http
GET /health
```

Checks whether the API is healthy and whether the model is loaded.

Example:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## Model Information

```http
GET /info
```

Returns model information.

Example:

```json
{
  "model_name": "Olist_Delay_Model",
  "model_version": "v1.0.0-onnx",
  "source": "ONNX Model"
}
```

---

## Required Features

```http
GET /features
```

Returns:

* Required categorical features
* Required numerical features
* Feature order
* Number of final encoded features

---

## Single Prediction

```http
POST /predict
```

Receives raw order information and performs:

```text
Raw Input
   ↓
Feature Engineering
   ↓
Imputation
   ↓
Encoding
   ↓
Scaling
   ↓
ONNX Inference
   ↓
Prediction
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.82,
  "model_version": "v1.0.0-onnx"
}
```

Where:

```text
prediction = 0 → predicted on-time delivery
prediction = 1 → predicted late delivery
```

---

## Batch Prediction

```http
POST /predict/batch
```

Allows multiple orders to be predicted in one request.

Example response:

```json
{
  "predictions": [
    {
      "prediction": 1,
      "probability": 0.82,
      "model_version": "v1.0.0-onnx"
    }
  ],
  "total_count": 1,
  "model_version": "v1.0.0-onnx"
}
```

---

#  Monitoring

The application includes monitoring using:

```text
FastAPI
    ↓
Prometheus
    ↓
Grafana
```

---

##  Service Metrics

The API exposes:

```http
GET /metrics
```

Prometheus instrumentation automatically tracks service metrics including:

* Request count
* Request latency
* HTTP status codes
* Errors

The FastAPI application uses:

```python
Instrumentator().instrument(app).expose(app)
```

---

#  Prediction Distribution

The service tracks prediction counts using a Prometheus Counter:

```text
model_predictions_total
```

Predictions are grouped by their output label:

```text
prediction="0"
prediction="1"
```

This allows us to monitor the prediction distribution over time.

Example:

```text
On-time predictions: 70%
Late predictions:    30%
```

A significant change from the normal baseline may indicate:

* Data distribution changes
* Changes in customer/order behavior
* Potential model drift
* Upstream data quality problems

---

#  Prediction Logging

Prediction results are logged for future evaluation.

For single predictions, the evaluation log contains:

* Order input
* Prediction
* Probability
* Model version

For batch predictions, the log contains:

* Batch size
* Prediction results

The purpose is to keep the prediction available until the **actual delivery date** becomes known.

Then the prediction can be compared with the real outcome:

```text
Prediction
     ↓
Wait for actual delivery
     ↓
Actual result
     ↓
Compare
     ↓
Evaluate model performance
```

---

#  Drift Monitoring

Prediction distribution can be monitored over time.

A baseline distribution is established from normal model behavior.

Example:

```text
Baseline:

Prediction 0 → 70%
Prediction 1 → 30%
```

If the production distribution changes significantly:

```text
Current:

Prediction 0 → 45%
Prediction 1 → 55%
```

this should trigger investigation for possible drift.

Drift monitoring should consider:

* Prediction distribution
* Input data distribution
* Missing-value rates
* Feature ranges
* Category distribution

---

#  Alerting Strategy

The following alert thresholds are proposed:

| Alert                   | Condition                           | Severity    |
| ----------------------- | ----------------------------------- | ----------- |
| API Error Rate          | > 5% for 5 minutes                  | 🔴 Critical |
| P95 Latency             | > 1 second                          | 🟡 Warning  |
| Prediction Drift        | Significant deviation from baseline | 🟡 Warning  |
| Severe Prediction Drift | Large deviation from baseline       | 🔴 Critical |
| Model Unavailable       | `/health` reports unhealthy         | 🔴 Critical |

These alerts help detect:

* API failures
* Performance degradation
* Model availability issues
* Unexpected prediction behavior
* Potential data/model drift

---

#  Docker

The application uses a **multi-stage Docker build**.

## Docker Architecture

```text
Stage 1: Builder
────────────────────────────
uv Python 3.13
       ↓
Install dependencies
       ↓
Build application environment
       ↓
/app/.venv


Stage 2: Runtime
────────────────────────────
Python 3.13 slim
       ↓
Copy .venv
       ↓
Copy API + required config
       ↓
Run as non-root user
       ↓
FastAPI
```

The builder uses:

```text
ghcr.io/astral-sh/uv:python3.13-bookworm-slim
```

The final runtime image uses:

```text
python:3.13-slim
```

The runtime image intentionally excludes unnecessary training files and development dependencies.

---

## 🔐 Docker Security

The application does not run as root.

A dedicated user is created:

```text
appuser
```

The API runs under this non-root user.

The Docker image also includes a health check:

```text
/health
```

---

##  Build Docker Image

```bash
docker build -t olist-delay-api:latest .
```

---

##  Run Docker Container

```bash
docker run -p 8000:8000 olist-delay-api:latest
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

Metrics:

```text
http://localhost:8000/metrics
```

---

#  Docker Compose

The project includes three services:

```text
┌─────────────────────┐
│       API           │
│ FastAPI + ONNX      │
│ Port 8000           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Prometheus       │
│    Port 9090        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      Grafana        │
│      Port 3000      │
└─────────────────────┘
```

Services:

### API

```text
FastAPI + ONNX
Port: 8000
```

### Prometheus

```text
Metrics collection
Port: 9090
```

### Grafana

```text
Monitoring dashboard
Port: 3000
```

---

#  Start the Monitoring Stack

Run:

```bash
docker compose up --build
```

Or run in background:

```bash
docker compose up --build -d
```

Check running services:

```bash
docker compose ps
```

---

#  Monitoring URLs

After starting Docker Compose:

### FastAPI

```text
http://localhost:8000
```

### Swagger

```text
http://localhost:8000/docs
```

### Prometheus

```text
http://localhost:9090
```

### Grafana

```text
http://localhost:3000
```

Grafana credentials:

```text
Username: admin
Password: admin
```

---

#  Prometheus Configuration

Prometheus scrapes the FastAPI metrics endpoint every 5 seconds.

Configuration:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: 'olist_fastapi_metrics'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['api:8000']
```

The Docker Compose network allows Prometheus to reach the API using:

```text
api:8000
```

---

#  Docker Volumes

The API uses the following volumes:

```text
./artifacts → /app/artifacts:ro
./models    → /app/models:ro
api_logs    → /app/logs
```

The model and artifacts are mounted read-only.

Application logs are stored in a Docker named volume.

---

#  Health Checks

The API exposes:

```http
GET /health
```

Docker uses this endpoint to determine whether the API service is healthy.

Prometheus waits for the API service to become healthy before starting metric collection.

---

#  CI/CD

GitHub Actions is used to automate project validation.

The CI pipeline includes automated testing and Docker workflow execution.

The project uses Docker Hub for container image publishing.

Docker authentication is performed using GitHub repository secrets:

```text
DOCKER_USERNAME
DOCKER_PASSWORD
```

`DOCKER_PASSWORD` should contain a Docker Hub **Personal Access Token** rather than a normal password.

---

#  Main Dependencies

The project defines its dependencies in `pyproject.toml`.

Core MLOps dependencies include:

```text
dvc
great-expectations
mlflow
prometheus-fastapi-instrumentator
```

Machine learning dependencies include:

```text
scikit-learn
xgboost
scipy
pandas
```

Model serving dependencies include:

```text
onnxruntime
onnxmltools
skl2onnx
joblib
```

Development dependencies include:

```text
pytest
ruff
pre-commit
```

---

#  Complete MLOps Workflow

```text
1. Data
   │
   ▼
2. DVC Versioning
   │
   ▼
3. Great Expectations Validation
   │
   ▼
4. Feature Engineering
   │
   ▼
5. Train Model
   │
   ▼
6. MLflow Experiment Tracking
   │
   ▼
7. Export Model to ONNX
   │
   ▼
8. FastAPI Serving
   │
   ▼
9. Docker Container
   │
   ▼
10. Prometheus Metrics
   │
   ▼
11. Grafana Dashboard
   │
   ▼
12. Prediction Logging
   │
   ▼
13. Drift Monitoring
   │
   ▼
14. Alerts
```

---

#  Verification Checklist

Before deployment, verify:

```text
☑ uv sync
☑ pytest
☑ ruff check
☑ Docker build
☑ Docker container starts
☑ /health returns healthy
☑ /docs is accessible
☑ /predict works
☑ /predict/batch works
☑ /metrics is accessible
☑ Prometheus receives metrics
☑ Grafana is accessible
☑ Prediction metrics are generated
```



Olist Late Delivery Prediction Pipeline — MLOps Project

GitHub:

https://github.com/shimaa83/Olist-Late-Delivery-Prediction-Pipeline_v2

