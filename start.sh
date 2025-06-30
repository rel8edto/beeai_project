#!/usr/bin/env bash
set -euo pipefail   # e = exit on error, u = undefined var error, o pipefail = fail on broken pipes

# ── Required environment variables ──────────────────────────────────────────────
required_vars=(
  WATSONX_APIKEY
  WATSONX_URL
  WATSONX_PROJECT_ID
  OCTAGON_API_KEY        # new one
)

for v in "${required_vars[@]}"; do
  : "${!v:?Environment variable '$v' is required but not set or empty}"
done

# ── Optional defaults ──────────────────────────────────────────────────────────
: "${PORT:=8080}"        # default to 8080 if unset/empty

echo "✓ All required environment variables present."
echo "▶️ Starting FastAPI on port $PORT …"

exec uvicorn api:app --host 0.0.0.0 --port "$PORT"
