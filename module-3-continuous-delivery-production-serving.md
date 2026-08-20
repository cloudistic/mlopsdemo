# Module 3: Continuous Delivery & Production Serving (30 Mins)

## Theory & Concepts

### 1) Inference Paradigms
When moving ML models into production, select the serving pattern based on latency, throughput, and operational constraints:

- **Batch Scoring**
  - Best for large, periodic workloads (hourly/daily).
  - Optimized for cost efficiency over low latency.
  - Typical use cases: churn scoring, nightly demand forecasts.

- **Synchronous APIs (REST/gRPC)**
  - Best for real-time, request/response interactions.
  - Prioritizes low-latency predictions (milliseconds to seconds).
  - Typical use cases: fraud checks at checkout, recommendation lookup.

- **Asynchronous Queues**
  - Best when workloads are bursty, long-running, or fault-tolerant.
  - Decouples request ingestion from model inference execution.
  - Typical use cases: document analysis, media processing, back-office scoring.

- **Streaming Inference**
  - Best for continuous event-driven predictions.
  - Integrates with stream processors and message buses.
  - Typical use cases: anomaly detection on telemetry, clickstream personalization.

**Selection Lens:**
- Latency target (real-time vs eventual)
- Throughput profile (steady vs bursty)
- Cost envelope (always-on vs scheduled)
- Reliability requirements (retries, replay, dead-letter handling)

### 2) CI/CD for ML (Continuous Integration / Continuous Training)
A robust MLOps delivery process should validate not only software correctness but also model quality and data assumptions:

- **Code Unit Tests**
  - Verify feature transformations, utility functions, and prediction wrappers.

- **Pipeline Integration Tests**
  - Validate end-to-end flow across ingestion, preprocessing, model loading, and inference endpoints.

- **Threshold / Regression Validations**
  - Compare candidate model metrics against baseline benchmarks.
  - Fail builds when quality drops below defined gates.

- **Continuous Training Hooks**
  - Trigger retraining on schedules or data drift signals.
  - Promote artifacts only after passing reproducibility and quality checks.

### 3) Safe Deployment Strategies
Deployment should minimize production risk while allowing rapid iteration:

- **Shadow Deployment**
  - Route a copy of live traffic to a new model without affecting user responses.
  - Compare outputs and latency before exposure.

- **Blue/Green Deployment**
  - Maintain two production environments and switch traffic atomically.
  - Enables fast rollback by reverting traffic to the stable environment.

- **Canary Deployment**
  - Release to a small traffic segment first.
  - Expand gradually if SLOs and model KPIs remain healthy.
  - Use automated rollback on error spikes, latency regressions, or metric degradation.

---

## Practical Example: Serving & Testing Pipeline

This demo wraps a serialized model in a FastAPI service with request validation, packages it in Docker, and validates behavior in GitHub Actions.

### A) FastAPI Service (with Pydantic validation)

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="MLOps Demo Inference Service")
model = joblib.load("artifacts/model.joblib")

class PredictRequest(BaseModel):
    feature_1: float = Field(..., description="First numerical feature")
    feature_2: float = Field(..., description="Second numerical feature")
    feature_3: float = Field(..., description="Third numerical feature")

class PredictResponse(BaseModel):
    prediction: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    x = np.array([[payload.feature_1, payload.feature_2, payload.feature_3]])
    y = model.predict(x)[0]
    return PredictResponse(prediction=float(y))
```

### B) Minimal Validation Tests

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_schema_and_status():
    payload = {
        "feature_1": 1.2,
        "feature_2": 3.4,
        "feature_3": 5.6,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)
```

### C) Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### D) GitHub Actions Workflow (run tests on every push)

```yaml
name: ci-ml-serving

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest -q
```

---

## Demo Walkthrough (Suggested Speaking Flow)
1. Explain why real-time API serving is chosen for this scenario.
2. Show the `/predict` contract and Pydantic input validation.
3. Run tests locally (`pytest`) to validate health and inference route behavior.
4. Build and run container:
   - `docker build -t mlopsdemo:module3 .`
   - `docker run -p 8000:8000 mlopsdemo:module3`
5. Explain CI checks in GitHub Actions and how they gate changes.
6. Close with rollout safety: start with shadow/canary before full production cutover.

## Learning Outcomes
By the end of this module, participants should be able to:
- Choose the right inference paradigm for a business and systems context.
- Implement a basic production inference service with schema validation.
- Containerize model-serving workloads for portable deployment.
- Configure CI checks that validate service behavior on every change.
- Articulate risk-reduction patterns for safe model rollouts.
