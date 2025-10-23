#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/dzoan/demo-library}"
STATE_FILE="$ROOT/.restoreok_state"

echo "🧩 Interactive Restore Utility (multi-version aware + remembers last choice)"
echo "Root: $ROOT"
echo "State: $STATE_FILE"
echo "───────────────────────────────────────────────"

# -------- Files to manage (update as needed) --------
FILES=(
  "$ROOT/services/ui/app.py"
  "$ROOT/services/api/main.py"
  "$ROOT/services/api/routers/agents.py"
  "$ROOT/services/api/routers/reports.py"
  "$ROOT/services/api/routers/settings.py"
  "$ROOT/services/api/routers/training.py"
  "$ROOT/services/api/routers/system.py"
  "$ROOT/services/api/routers/export.py"
  "$ROOT/services/api/routers/runs.py"
  "$ROOT/services/api/routers/admin.py"
  "$ROOT/agents/credit_appraisal/agent.py"
  "$ROOT/agents/credit_appraisal/model_utils.py"
  "$ROOT/agent_platform/agent_sdk/sdk.py"
  "$ROOT/scripts/generate_training_dataset.py"
  "$ROOT/scripts/run_e2e.sh"
  "$ROOT/infra/run_api.sh"
  "$ROOT/tests/test_api_e2e.py"
)

# -------- State helpers --------
declare -A LAST_CHOICE  # filepath -> backup_path

load_state() {
  [[ -f "$STATE_FILE" ]] || return 0
  while IFS='' read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    # expected format: <filepath>|<backup_path>
    IFS='|' read -r fpath bpath <<<"$line" || true
    [[ -n "${fpath:-}" && -n "${bpath:-}" ]] && LAST_CHOICE["$fpath"]="$bpath"
  done <"$STATE_FILE"
}

save_state() {
  local tmp="$STATE_FILE.tmp"
  : >"$tmp"  # truncate
  # Keep only entries for files we care about and that still exist
  for f in "${!LAST_CHOICE[@]}"; do
    echo "$f|${LAST_CHOICE[$f]}" >>"$tmp"
  done
  mv -f "$tmp" "$STATE_FILE"
}

record_choice() {
  local file="$1"
  local selected="$2"
  LAST_CHOICE["$file"]="$selected"
}

# -------- Main --------
load_state

RESTORED=0
SKIPPED=0
MISSING=0

for file in "${FILES[@]}"; do
  echo
  echo "───────────────────────────────────────────────"
  echo "📄 File: $file"

  if [[ ! -f "$file" ]]; then
    echo "⚠️  Original file does not exist (you can still restore it if backups exist)."
  fi

  # List backups
  mapfile -t matches < <(ls -1 "${file}".ok.*.bak 2>/dev/null || true)

  if (( ${#matches[@]} == 0 )); then
    echo "⚠️  No backups found."
    (( ++MISSING ))
    continue
  fi

  echo "🗂️  Available backups:"
  local last_idx="" last_display=""
  for i in "${!matches[@]}"; do
    local idx=$((i+1))
    local tag=""
    if [[ -n "${LAST_CHOICE[$file]:-}" && "${LAST_CHOICE[$file]}" == "${matches[$i]}" ]]; then
      tag="  ⟵ last used"
      last_idx="$idx"
      last_display="${matches[$i]}"
    fi
    echo "  [$idx] ${matches[$i]}$tag"
  done

  # Prompt
  if [[ -n "$last_idx" ]]; then
    echo -n "Select version to restore [1-${#matches[@]}], Enter = use last ($last_idx), 0 = skip: "
  else
    echo -n "Select version to restore [1-${#matches[@]}], Enter = skip, 0 = skip: "
  fi
  read -r choice

  # Interpret choice
  if [[ -z "$choice" ]]; then
    if [[ -n "$last_idx" ]]; then
      choice="$last_idx"
      echo "→ Using last choice: #$choice ($last_display)"
    else
      echo "⏭️  Skipped $file"
      (( ++SKIPPED ))
      continue
    fi
  fi

  if ! [[ "$choice" =~ ^[0-9]+$ ]]; then
    echo "❌ Invalid input. Skipping."
    (( ++SKIPPED ))
    continue
  fi

  if (( choice == 0 )); then
    echo "⏭️  Skipped $file"
    (( ++SKIPPED ))
    continue
  fi

  if (( choice < 1 || choice > ${#matches[@]} )); then
    echo "❌ Out of range. Skipping."
    (( ++SKIPPED ))
    continue
  fi

  selected="${matches[$((choice-1))]}"

  # Confirm & restore
  echo "🔁 Restoring: $(basename "$file") ← $(basename "$selected")"
  mkdir -p "$(dirname "$file")"
  cp -f "$selected" "$file"
  echo "✅ Restored → $file"
  record_choice "$file" "$selected"
  (( ++RESTORED ))
done

save_state

echo
echo "🎯 Restore Summary:"
echo "   ✅ Restored : $RESTORED file(s)"
echo "   ⏭️  Skipped : $SKIPPED file(s)"
echo "   ⚠️  Missing : $MISSING file(s)"
echo "State saved to: $STATE_FILE"
echo "───────────────────────────────────────────────"

