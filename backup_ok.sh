#!/usr/bin/env bash
set -euo pipefail

BACKUP_EXT=".ok.$(date +%Y%m%d-%H%M%S).bak"

FILES=(
  "/home/dzoan/demo-library/services/ui/app.py"
  "/home/dzoan/demo-library/services/ui/requirements.txt"
  "/home/dzoan/demo-library/services/ui/runwebui.sh"

  "/home/dzoan/demo-library/services/api/routers/agents.py"
  "/home/dzoan/demo-library/services/api/routers/reports.py"
  "/home/dzoan/demo-library/services/api/routers/settings.py"
  "/home/dzoan/demo-library/services/api/routers/training.py"
  "/home/dzoan/demo-library/services/api/routers/system.py"
  "/home/dzoan/demo-library/services/api/routers/export.py"
  "/home/dzoan/demo-library/services/api/routers/runs.py"
  "/home/dzoan/demo-library/services/api/routers/admin.py"

  "/home/dzoan/demo-library/services/api/main.py"
  "/home/dzoan/demo-library/services/api/requirements.txt"
  "/home/dzoan/demo-library/services/api/adapters/__init__.py"
  "/home/dzoan/demo-library/services/api/adapters/llm_adapters.py"

  "/home/dzoan/demo-library/agents/credit_appraisal/agent.py"
  "/home/dzoan/demo-library/agents/credit_appraisal/model_utils.py"
  "/home/dzoan/demo-library/agents/credit_appraisal/__init__.py"
  "/home/dzoan/demo-library/agents/credit_appraisal/agent.yaml"
  "/home/dzoan/demo-library/agents/credit_appraisal/models/production/meta.json"

  "/home/dzoan/demo-library/agent_platform/agent_sdk/__init__.py"
  "/home/dzoan/demo-library/agent_platform/agent_sdk/sdk.py"

  "/home/dzoan/demo-library/services/train/train_credit.py"
  "/home/dzoan/demo-library/scripts/generate_training_dataset.py"
  "/home/dzoan/demo-library/scripts/run_e2e.sh"
  "/home/dzoan/demo-library/infra/run_api.sh"
  "/home/dzoan/demo-library/Makefile"
  "/home/dzoan/demo-library/pyproject.toml"

  "/home/dzoan/demo-library/tests/test_api_e2e.py"
  "/home/dzoan/demo-library/samples/credit/schema.json"
  "/home/dzoan/demo-library/agents/credit_appraisal/sample_data/credit_sample.csv"
  "/home/dzoan/demo-library/agents/credit_appraisal/sample_data/credit_training_sample.csv"
)

echo "==> Backing up important files under: /home/dzoan/demo-library"
echo "==> Backup suffix: ${BACKUP_EXT}"
echo

missing=0
declare -a EXISTING=()
for f in "${FILES[@]}"; do
  if [[ -f "$f" ]]; then
    echo "  • $f"
    EXISTING+=("$f")
  else
    echo "  • $f   (skip: not found)"
    ((missing++)) || true
  fi
done

if (( ${#EXISTING[@]} == 0 )); then
  echo "❌ None of the listed files exist. Exiting."
  exit 1
fi

echo
if (( missing > 0 )); then
  echo "⚠️  $missing file(s) were not found and will be skipped."
fi
echo

read -p "Proceed with backup of the existing files listed above? [y/N] " -n 1 -r
echo
[[ $REPLY =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

BACKUP_COUNT=0
SKIPPED_COUNT=0
SUDO_BIN="$(command -v sudo || true)"

copy_inplace() {
  local src="$1"
  local dst="$2"
  local dir
  dir="$(dirname "$dst")"

  # If destination directory isn’t writable by current user, try sudo.
  if [[ -w "$dir" ]]; then
    cp -f "$src" "$dst"
  else
    if [[ -n "$SUDO_BIN" ]]; then
      echo "   (no write permission — using sudo)"
      $SUDO_BIN cp -f "$src" "$dst"
    else
      echo "   ❌ Cannot write to $dir and sudo not available — skipping."
      return 1
    fi
  fi
  return 0
}

echo
for file in "${EXISTING[@]}"; do
  bak="${file}${BACKUP_EXT}"
  echo "────────────────────────────────────────────"
  echo "➡️  Processing: $file"

  if [[ -f "$bak" ]]; then
    echo "   ⚠️  Backup already exists: $bak"
    read -p "   Overwrite it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      if copy_inplace "$file" "$bak"; then
        echo "   ✅ Overwritten → $bak"
        ((BACKUP_COUNT++)) || true
      else
        echo "   ⏭️  Skipped (write failed)"
        ((SKIPPED_COUNT++)) || true
      fi
    else
      echo "   ⏭️  Skipped $file"
      ((SKIPPED_COUNT++)) || true
    fi
  else
    read -p "   Create new backup? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      if copy_inplace "$file" "$bak"; then
        echo "   ✅ Backed up → $bak"
        ((BACKUP_COUNT++)) || true
      else
        echo "   ⏭️  Skipped (write failed)"
        ((SKIPPED_COUNT++)) || true
      fi
    else
      echo "   ⏭️  Skipped $file"
      ((SKIPPED_COUNT++)) || true
    fi
  fi
done

echo "────────────────────────────────────────────"
echo "🎯 Backup complete — $BACKUP_COUNT file(s) backed up, $SKIPPED_COUNT skipped."
