#!/bin/bash

mkdir -p ./mlruns

# Set permissions to allow write access
chmod -R 777 ./mlruns

# Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
