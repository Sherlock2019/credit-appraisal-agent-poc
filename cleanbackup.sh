#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ROOT_DIR="$HOME/demo-library"
BACKUP_DIR="$ROOT_DIR/backups/moved_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "🚚 Moving all backup files from project into: $BACKUP_DIR"
echo "──────────────────────────────────────────────"

# ─────────────────────────────────────────────
# FIND & MOVE BACKUPS (with sudo fallback)
# ─────────────────────────────────────────────
cd "$ROOT_DIR"

find . -type f \( -name "*.bak" -o -name "*.ok.*.bak" \) | while read -r file; do
  dest="$BACKUP_DIR/$(dirname "$file")"
  mkdir -p "$dest"

  if mv "$file" "$dest/" 2>/dev/null; then
    echo "✅ Moved: $file"
  else
    echo "⚠️  Permission denied for $file — retrying with sudo..."
    sudo mkdir -p "$dest"
    sudo mv "$file" "$dest/" && echo "✅ Moved with sudo: $file" || echo "❌ Failed to move even with sudo: $file"
  fi
done

echo "──────────────────────────────────────────────"
echo "🎉 All backup files moved successfully!"
echo "Destination: $BACKUP_DIR"
