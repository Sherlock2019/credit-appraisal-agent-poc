#!/bin/bash
# restart_poc.sh — restart your Credit Appraisal PoC + Swagger UI

cd ~/demo-library || exit 1
source services/api/.venv/bin/activate
export PYTHONPATH=$(pwd)
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=phi3

# kill old uvicorn processes
echo "Stopping existing FastAPI servers..."
pkill -f "uvicorn" 2>/dev/null

echo "Starting Credit Appraisal API on port 8090..."
nohup python -m uvicorn services.api.main:app \
  --host 0.0.0.0 --port 8090 --reload > poc.log 2>&1 &

sleep 2
echo "✅ API restarted. Open Swagger UI at: http://localhost:8090/docs"
