# Module 1 Demo — Data Quality Gating

This demo implements a runtime data quality gate using **Pandera**.
The gate blocks pipeline execution when incoming data violates schema or drift thresholds.

## What is validated

- Required columns and data types
- Null checks
- Value ranges
- Basic drift checks against baseline means

## Project structure

- `demo_data_quality_gating.py` — main script
- `data/baseline_train_sample.csv` — baseline statistics reference
- `data/incoming_good.csv` — valid incoming payload
- `data/incoming_bad.csv` — invalid payload

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run with valid payload

```bash
python module1/demo_data_quality_gating.py \
  --baseline module1/data/baseline_train_sample.csv \
  --incoming module1/data/incoming_good.csv
```

Expected result: validation passes and pipeline proceeds.

## Run with invalid payload

```bash
python module1/demo_data_quality_gating.py \
  --baseline module1/data/baseline_train_sample.csv \
  --incoming module1/data/incoming_bad.csv
```

Expected result: validation fails and pipeline stops.

## Suggested 10-minute live demo flow

1. Explain train/serve skew in 1 minute.
2. Show the schema and drift thresholds in code.
3. Run good payload (pass).
4. Run bad payload (fail).
5. Explain how this gate integrates before training or serving steps in production pipelines.
