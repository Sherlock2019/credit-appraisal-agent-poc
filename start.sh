#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.venv"
LOGDIR="${ROOT}/.logs"
APIPORT="${APIPORT:-8090}"
UIPORT="${UIPORT:-8502}"

mkdir -p "$LOGDIR" "${ROOT}/services/api/.runs" "${ROOT}/agents/credit_appraisal/models/production"

# ── Ensure venv
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1090
source "${VENV}/bin/activate"

python -V
pip -V

# ── Install/upgrade deps for API + UI (idempotent)
python -m pip install -U pip wheel
pip install -r "${ROOT}/services/api/requirements.txt"
pip install -r "${ROOT}/services/ui/requirements.txt"

# ── Helpful for local package imports (agents/, services/, etc.)
export PYTHONPATH="${ROOT}"

# ── Start API (Uvicorn)
if [[ -f "${ROOT}/.pids/api.pid" ]] && kill -0 "$(cat "${ROOT}/.pids/api.pid")" 2>/dev/null; then
  echo "API already running (PID $(cat "${ROOT}/.pids/api.pid"))."
else
  mkdir -p "${ROOT}/.pids"
  nohup "${VENV}/bin/uvicorn" services.api.main:app \
        --host 0.0.0.0 --port "${APIPORT}" --reload \
        > "${LOGDIR}/api.log" 2>&1 &
  echo $! > "${ROOT}/.pids/api.pid"
  echo "✅ API started (PID=$(cat "${ROOT}/.pids/api.pid")) | log: ${LOGDIR}/api.log"
fi

# ── Start UI (Streamlit)
if [[ -f "${ROOT}/.pids/ui.pid" ]] && kill -0 "$(cat "${ROOT}/.pids/ui.pid")" 2>/dev/null; then
  echo "UI already running (PID $(cat "${ROOT}/.pids/ui.pid"))."
else
  mkdir -p "${ROOT}/.pids"
  nohup "${VENV}/bin/streamlit" run "${ROOT}/services/ui/app.py" \
        --server.port "${UIPORT}" --server.address 0.0.0.0 \
        > "${LOGDIR}/ui.log" 2>&1 &
  echo $! > "${ROOT}/.pids/ui.pid"
  echo "✅ UI started (PID=$(cat "${ROOT}/.pids/ui.pid")) | log: ${LOGDIR}/ui.log"
fi

echo "----------------------------------------------------"
echo "🎯 All services running in background!"
echo "📘 Swagger: http://localhost:${APIPORT}/docs"
echo "🌐 Web UI:  http://localhost:${UIPORT}"
echo "📂 Logs:    ${LOGDIR}"
echo "----------------------------------------------------"

# ── Background full log monitor
echo "🧩 Starting background log monitor (full logs)..."

nohup bash -c "tail -n 0 -f '${LOGDIR}/api.log' '${LOGDIR}/ui.log' \
  >> '${LOGDIR}/live_combined.log'" >/dev/null 2>&1 &

LOG_MONITOR_PID=$!
echo $LOG_MONITOR_PID > "${ROOT}/.pids/logmonitor.pid"

echo "✅ Live log monitor running in background (PID=${LOG_MONITOR_PID})"
echo "📄 Combined live output → ${LOGDIR}/live_combined.log"
