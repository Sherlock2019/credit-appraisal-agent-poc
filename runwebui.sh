#!/bin/bash
# ─────────────────────────────────────────────
# Simple launcher for the local Web UI
# Runs Streamlit UI using your FastAPI backend
# Port 8502 by default (avoids Docker conflict)
# ─────────────────────────────────────────────

set -e

echo "🌐 Starting AI Credit Appraisal Web UI..."

# Ensure we’re in the correct repo
cd ~/demo-library/services/api

# Activate virtual environment
if [ ! -d ".venv" ]; then
  echo "❌ Virtual environment not found in services/api/.venv"
  exit 1
fi
source .venv/bin/activate

# Install missing dependencies
pip install -q streamlit requests pandas

# Run the UI on port 8502 (avoids conflict with Docker 8501)
cd ~/demo-library/services/ui

echo "🚀 Opening Streamlit UI at http://localhost:8502"
streamlit run app.py --server.port 8502
