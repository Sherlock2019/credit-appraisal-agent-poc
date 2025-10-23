#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDDIR="${ROOT}/.pids"

echo "----------------------------------------------------"
echo "🛑 Stopping all background services..."
echo "----------------------------------------------------"

stop_service() {
  local name=$1
  local pid_file="${PIDDIR}/${name}.pid"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping ${name} (PID=${pid})..."
      kill "$pid" 2>/dev/null || true
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        echo "Force killing ${name} (PID=${pid})..."
        kill -9 "$pid" 2>/dev/null || true
      fi
      echo "✅ ${name} stopped."
    else
      echo "⚠️  ${name} PID ${pid} not running."
    fi
    rm -f "$pid_file"
  else
    echo "⚠️  No PID file for ${name} — already stopped or never started."
  fi
}

stop_service "api"
stop_service "ui"
stop_service "logmonitor"

echo "----------------------------------------------------"
echo "🧹 Cleanup done. All services stopped."
echo "----------------------------------------------------"

