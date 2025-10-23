#!/usr/bin/env bash
#
# restore_ok.sh — Restore from any ".ok.bak" snapshot.
# It lists all ok-backups and lets you choose a file to restore.
# Creates a safety pre-restore ".pre-<timestamp>.bak" of current file.
#
# Usage:
#   ./restore_ok.sh                    # interactive list of all *.ok.bak in repo
#   ./restore_ok.sh path/to/file       # list only backups for a specific file
#
set -euo pipefail

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Collect candidate backups
query="${1:-}"

if [[ -n "$query" ]]; then
  # Strip any existing .ok.bak suffix from query to normalize
  base_query="${query%%.ok.bak}"
  # Find ok backups that start with the same base filename
  mapfile -t backups < <(find . -type f -name "$(basename "$base_query")".*.ok.bak -o -path "*$(dirname "$base_query")/*$(basename "$base_query")".*.ok.bak | sort)
else
  mapfile -t backups < <(find . -type f -name "*.ok.bak" | sort)
fi

if [[ ${#backups[@]} -eq 0 ]]; then
  echo "No .ok.bak backups found."
  exit 0
fi

echo "Found ${#backups[@]} OK backups:"
printf "  [%3s]  %s\n" "#" "PATH"
for i in "${!backups[@]}"; do
  printf "  [%3d]  %s\n" "$i" "${backups[$i]}"
done
echo

read -rp "Enter the index to restore: " idx
if ! [[ "$idx" =~ ^[0-9]+$ ]] || (( idx < 0 || idx >= ${#backups[@]} )); then
  echo "Invalid selection."
  exit 1
fi

bak_path="${backups[$idx]}"
# Derive original path (strip the trailing .<timestamp>.ok.bak)
orig_path="${bak_path%.*.ok.bak}"

if [[ ! -f "$bak_path" ]]; then
  echo "Backup not found: $bak_path"
  exit 1
fi

# Ensure destination directory exists
dest_dir="$(dirname "$orig_path")"
mkdir -p -- "$dest_dir"

# If original exists, save a pre-restore backup
if [[ -f "$orig_path" ]]; then
  pre="${orig_path}.pre-${timestamp}.bak"
  cp -p -- "$orig_path" "$pre"
  echo "Saved pre-restore backup: $pre"
fi

# Restore
cp -p -- "$bak_path" "$orig_path"

echo "Restored:"
echo "  from: $bak_path"
echo "    to: $orig_path"

# Show diff summary if git is present
if command -v git >/dev/null 2>&1; then
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo
    echo "Git status (post-restore):"
    git status --short -- "$orig_path" || true
  fi
fi

echo "Done."

