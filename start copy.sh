#!/usr/bin/env bash
set -e

# Optional sanity checks
: "${WATSONX_APIKEY:?Missing WATSONX_APIKEY}"
: "${WATSONX_URL:?Missing WATSONX_URL}"
: "${WATSONX_PROJECT_ID:?Missing WATSONX_PROJECT_ID}"

# Run the FastAPI service
# exec uv run server --host 0.0.0.0 --port "${PORT:=8080}"
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:=8080}"

