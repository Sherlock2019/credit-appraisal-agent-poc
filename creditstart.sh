#!/usr/bin/env bash
set -euo pipefail

# ╭──────────────────────────────────────────────────────────╮
# │ Credit Appraisal POC launcher (API + Streamlit UI)       │
# │ - Frees ports 8090 (API) and 8502 (UI) before starting   │
# │ - Creates/uses venv, installs deps, starts services      │
# ╰──────────────────────────────────────────────────────────╯

# ─────────────────────────────────────────
# Config
ROOT="$HOME/demo-library"
API_DIR="$ROOT/services/api"
UI_DIR="$ROOT/services/ui"
VENV_DIR="$API_DIR/.venv"

DEFAULT_API_PORT=8090
DEFAULT_UI_PORT=8502
PYBIN="python3"

# ─────────────────────────────────────────
# Utilities

# Kill any process bound to a TCP port (multi-PID, sudo fallback, retries)
kill_port() {
  local port="$1"
  local tries=5
  echo "🧹 Ensuring port $port is free…"

  while (( tries-- > 0 )); do
    # Collect all listeners' PIDs (prefer lsof, then ss, then fuser)
    mapfile -t PIDS < <(
      { command -v lsof >/dev/null && lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null; } ||
      { command -v ss   >/dev/null && ss -ltnp "( sport = :$port )" 2>/dev/null \
          | awk -F 'pid=' '/pid=/{split($2,a,","); print a[1]}' | sort -u; } ||
      { command -v fuser >/dev/null && fuser "$port"/tcp 2>/dev/null | tr ' ' '\n'; } ||
      true
    )

    if [[ ${#PIDS[@]} -eq 0 || -z "${PIDS[*]:-}" ]]; then
      echo "✅ Port $port is free."
      return 0
    fi

    echo "⚠️  Killing PID(s) on $port: ${PIDS[*]}"
    kill "${PIDS[@]}" 2>/dev/null || true
    sleep 0.4

    # Escalate if still busy
    if command -v ss >/dev/null && ss -ltnp "( sport = :$port )" 2>/dev/null | grep -q ":$port"; then
      echo "🔒 Elevated kill on $port…"
      if command -v sudo >/dev/null 2>&1; then
        sudo kill -9 "${PIDS[@]}" 2>/dev/null || true
        command -v fuser >/dev/null 2>&1 && sudo fuser -k "$port"/tcp 2>/dev/null || true
      fi
      sleep 0.6
    fi

    # Re-check
    if ! (command -v ss >/dev/null && ss -ltnp "( sport = :$port )" 2>/dev/null | grep -q ":$port"); then
      echo "✅ Port $port cleared."
      return 0
    fi
    echo "…still busy; retrying"
  done

  echo "❌ Could not free port $port after multiple attempts."
  return 1
}

# Check if a TCP port is in use
port_in_use() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "( sport = :$port )" | grep -q ":$port" || return 1
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP -sTCP:LISTEN -P | awk '{print $9}' | grep -q ":$port" || return 1
  else
    (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1 && return 0 || return 1
  fi
  return 0
}

# Return the next free port starting at $1 (tries up to +20)
next_free_port() {
  local start="$1"
  local p="$start"
  for _ in $(seq 0 20); do
    if port_in_use "$p"; then
      p=$((p+1))
    else
      echo "$p"
      return 0
    fi
  done
  echo "$start"
  return 0
}

# Wait for HTTP endpoint to respond (uses curl or python)
wait_for_http() {
  local url="$1"
  local attempts="${2:-50}"
  local delay_s="${3:-0.2}"
  local pid_check="${4:-}"  # optional PID to ensure the process didn't die

  echo -n "⏳ Waiting for $url"
  for _ in $(seq 1 "$attempts"); do
    if [[ -n "$pid_check" ]] && ! ps -p "$pid_check" >/dev/null 2>&1; then
      echo
      echo "❌ Process $pid_check died during startup. Last logs:"
      tail -n 200 /tmp/creditpoc_api.log || true
      return 2
    fi

    if command -v curl >/dev/null 2>&1; then
      if curl -fsS "$url" >/dev/null 2>&1; then echo " ready."; return 0; fi
    else
      # python fallback
      $PYBIN - <<PY >/dev/null 2>&1 && { echo " ready."; return 0; }
import urllib.request
urllib.request.urlopen("$url", timeout=1)
PY
    fi
    sleep "$delay_s"
    echo -n "."
  done
  echo
  echo "⚠️  Timeout waiting for $url"
  return 1
}

# ─────────────────────────────────────────
echo "🧪 Credit POC start…"

# Sanity checks
[[ -d "$ROOT" ]]    || { echo "❌ Project root not found at $ROOT"; exit 1; }
[[ -d "$API_DIR" ]] || { echo "❌ API dir not found at $API_DIR"; exit 1; }
[[ -d "$UI_DIR" ]]  || { echo "❌ UI dir not found at $UI_DIR"; exit 1; }
command -v "$PYBIN" >/dev/null 2>&1 || { echo "❌ python3 not found"; exit 1; }

# Free default ports up-front (hard requirement)
kill_port "$DEFAULT_API_PORT"   # 8090
kill_port "$DEFAULT_UI_PORT"    # 8502

# Python & venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "🌱 Creating venv at $VENV_DIR"
  "$PYBIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "⬆️  Upgrading pip"
python -m pip install --upgrade pip -q

# Install API deps
if [[ -f "$API_DIR/requirements.txt" ]]; then
  echo "📦 Installing API requirements.txt"
  pip install -r "$API_DIR/requirements.txt" -q || true
fi

# Ensure core deps present for API & UI
echo "📦 Ensuring core dependencies"
pip install fastapi uvicorn pandas numpy requests joblib shap fpdf scikit-learn lightgbm streamlit -q || true

# Export PYTHONPATH so modules resolve
export PYTHONPATH="$ROOT"

# Pick ports (in case something respawns 8090/8502 after we killed them)
API_PORT="$(next_free_port "$DEFAULT_API_PORT")"
UI_PORT="$(next_free_port "$DEFAULT_UI_PORT")"

if [[ "$API_PORT" != "$DEFAULT_API_PORT" ]]; then
  echo "ℹ️  API default port $DEFAULT_API_PORT busy — using $API_PORT"
fi
if [[ "$UI_PORT" != "$DEFAULT_UI_PORT" ]]; then
  echo "ℹ️  UI default port $DEFAULT_UI_PORT busy — using $UI_PORT"
fi

# ─────────────────────────────────────────
# Start API
echo "🚀 Starting API (FastAPI/Uvicorn) on :$API_PORT"
cd "$ROOT"
python -m uvicorn services.api.main:app \
  --host 0.0.0.0 --port "$API_PORT" --reload \
  >/tmp/creditpoc_api.log 2>&1 &
API_PID=$!

sleep 0.5
if ! ps -p "$API_PID" >/dev/null 2>&1; then
  echo "❌ API process died immediately. Last logs:"
  tail -n 200 /tmp/creditpoc_api.log || true
  exit 1
fi

# ✅ Wait on health (faster/more robust than /docs)
wait_for_http "http://127.0.0.1:$API_PORT/v1/health" 60 0.5 "$API_PID" || {
  echo "⚠️  API health check failed. Showing last logs then continuing so you can inspect UI."
  tail -n 200 /tmp/creditpoc_api.log || true
}

echo "📜 Swagger: http://localhost:$API_PORT/docs"
echo "📝 API logs: tail -f /tmp/creditpoc_api.log"

# Ensure cleanup on exit
cleanup() {
  echo "🧹 Stopping API (pid $API_PID)"
  if ps -p "$API_PID" >/dev/null 2>&1; then
    kill "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ─────────────────────────────────────────
# Start UI (foreground)
# 👉 Pass API_URL so the UI knows where to call
export API_URL="http://localhost:${API_PORT}"

echo "🖥️  Starting Web UI (Streamlit) on :$UI_PORT (API_URL=$API_URL)"
cd "$UI_DIR"
exec env API_URL="$API_URL" streamlit run app.py --server.port "$UI_PORT"

