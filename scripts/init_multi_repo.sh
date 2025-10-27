#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$HOME/rackspace-aisandbox"
HUB_REPO="${HUB_REPO:-https://github.com/Sherlock2019/AI-AIGENTbythePeoplesANDBOX.git}"
HUB_REF="${HUB_REF:-main}"
CREDIT_REPO="${CREDIT_REPO:-https://github.com/Sherlock2019/credit-appraisal-agent-poc.git}"
CREDIT_REF="${CREDIT_REF:-main}"
ASSET_REPO="${ASSET_REPO:-https://github.com/Sherlock2019/asset-appraisal-agent-.git}"
ASSET_REF="${ASSET_REF:-main}"
SHARED_SDK_REPO="${SHARED_SDK_REPO:-}"
SHARED_SDK_REF="${SHARED_SDK_REF:-main}"
SEED_HUB="true"

usage() {
  local exit_code="${1:-0}"
  cat <<'USAGE'
Usage: init_multi_repo.sh [OPTIONS] [WORKSPACE]

Bootstrap the AI-AIGENTbythePeoplesANDBOX multi-repo workspace and wire the
credit / asset appraisal agents as git submodules.

Options:
  -w, --workspace PATH     Target workspace directory (default: ~/rackspace-aisandbox)
      --hub-ref REF         Checkout REF for the orchestrator repository (default: main)
      --credit-ref REF      Checkout REF for the credit agent submodule (default: main)
      --asset-ref REF       Checkout REF for the asset agent submodule (default: main)
      --shared-sdk-repo URL Optional shared SDK repository to add as a submodule
      --shared-sdk-ref REF  Branch or tag for the shared SDK (default: main)
      --skip-seed           Do not scaffold FastAPI boilerplate inside the hub repo
  -h, --help                Show this help message and exit

Environment variables HUB_REPO, CREDIT_REPO, ASSET_REPO, and SHARED_SDK_REPO can
override the default git remotes when set.
USAGE
  exit "$exit_code"
}

POSITIONAL_SEEN=false
while [ $# -gt 0 ]; do
  case "$1" in
    -w|--workspace)
      ROOT_DIR="$2"
      shift 2
      ;;
    --hub-ref)
      HUB_REF="$2"
      shift 2
      ;;
    --credit-ref)
      CREDIT_REF="$2"
      shift 2
      ;;
    --asset-ref)
      ASSET_REF="$2"
      shift 2
      ;;
    --shared-sdk-repo)
      SHARED_SDK_REPO="$2"
      shift 2
      ;;
    --shared-sdk-ref)
      SHARED_SDK_REF="$2"
      shift 2
      ;;
    --skip-seed)
      SEED_HUB="false"
      shift
      ;;
    -h|--help)
      usage
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage 1
      ;;
    *)
      if [ "$POSITIONAL_SEEN" = false ]; then
        ROOT_DIR="$1"
        POSITIONAL_SEEN=true
        shift
      else
        echo "Unexpected argument: $1" >&2
        usage 1
      fi
      ;;
  esac
done

log() {
  printf "[init-multi-repo] %s\n" "$*"
}

ensure_git_repo() {
  local path="$1"
  local remote="$2"
  local ref="$3"
  if [ ! -d "$path/.git" ]; then
    log "Initialising orchestrator repo in $path"
    mkdir -p "$path"
    (cd "$path" && git init >/dev/null)
  fi
  if [ -n "$remote" ]; then
    (cd "$path" && {
      if git remote get-url origin >/dev/null 2>&1; then
        git remote set-url origin "$remote"
      else
        git remote add origin "$remote"
      fi
      git fetch origin "$ref" >/dev/null 2>&1 || true
      local current_branch
      current_branch=$(git symbolic-ref --quiet --short HEAD 2>/dev/null || echo "")
      if [ -n "$ref" ] && [ "$current_branch" != "$ref" ]; then
        git checkout "$ref" 2>/dev/null || git checkout -b "$ref" "origin/$ref" 2>/dev/null || true
      fi
    })
  fi
}

add_or_update_submodule() {
  local repo_url="$1"
  local sub_path="$2"
  local ref="$3"
  if [ -z "$repo_url" ]; then
    return
  fi
  local sub_name
  sub_name="$(echo "$sub_path" | tr '/' '-')"
  local module_key
  module_key=$(git config --file .gitmodules --get-regexp "^submodule\\..*\\.path$" | awk -v path="$sub_path" '$2==path {print $1}' | head -n1)
  if git config --file .gitmodules --get-regexp "^submodule\\.${sub_path//\//\\.}\\.path$" >/dev/null 2>&1; then
    log "Updating submodule $sub_path"
    git submodule sync -- "$sub_path"
    if [ -n "$ref" ] && [ -n "$module_key" ]; then
      git config -f .gitmodules "$module_key.branch" "$ref"
    fi
    git submodule update --init --remote "$sub_path"
    if [ -n "$ref" ]; then
      git -C "$sub_path" fetch origin "$ref" >/dev/null 2>&1 || true
      git -C "$sub_path" checkout "$ref" 2>/dev/null || git -C "$sub_path" checkout -B "$ref" "origin/$ref" 2>/dev/null || true
    fi
  else
    log "Adding submodule $sub_path"
    if [ -n "$ref" ]; then
      git submodule add --force -b "$ref" --name "$sub_name" "$repo_url" "$sub_path"
    else
      git submodule add --force --name "$sub_name" "$repo_url" "$sub_path"
    fi
    if [ -n "$ref" ]; then
      git config -f .gitmodules "submodule.$sub_name.branch" "$ref"
    fi
  fi
}

seed_file() {
  local target="$1"
  local payload="$2"
  if [ ! -f "$target" ]; then
    log "Creating $target"
    mkdir -p "$(dirname "$target")"
    printf "%s" "$payload" > "$target"
  fi
}

mkdir -p "$ROOT_DIR"
log "Using workspace $ROOT_DIR"

# Clone orchestrator repo if not present
if [ ! -d "$ROOT_DIR/AI-AIGENTbythePeoplesANDBOX" ]; then
  log "Cloning orchestrator repo"
  git clone --origin origin "$HUB_REPO" "$ROOT_DIR/AI-AIGENTbythePeoplesANDBOX"
else
  log "Refreshing orchestrator repo"
  (cd "$ROOT_DIR/AI-AIGENTbythePeoplesANDBOX" && {
    git fetch origin "$HUB_REF" >/dev/null 2>&1 || git fetch origin >/dev/null 2>&1 || true
    git merge --ff-only "origin/$HUB_REF" >/dev/null 2>&1 || true
  })
fi

HUB_PATH="$ROOT_DIR/AI-AIGENTbythePeoplesANDBOX"
ensure_git_repo "$HUB_PATH" "$HUB_REPO" "$HUB_REF"

cd "$HUB_PATH"

# Submodules for each agent
add_or_update_submodule "$CREDIT_REPO" "agent-credit-appraisal" "$CREDIT_REF"
add_or_update_submodule "$ASSET_REPO" "agent-asset-appraisal" "$ASSET_REF"
if [ -n "$SHARED_SDK_REPO" ]; then
  add_or_update_submodule "$SHARED_SDK_REPO" "shared-agent-sdk" "$SHARED_SDK_REF"
fi

# Seed boilerplate hub files when missing
if [ "$SEED_HUB" = "true" ] && [ ! -s "services/api/main.py" ]; then
  log "Seeding FastAPI hub app"
  mkdir -p services/api
  cat <<'PY' > services/api/main.py
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
import httpx
import yaml

app = FastAPI(title="AI Agent Hub")
registry: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
def load_registry() -> None:
    global registry
    with open("agent_registry.yaml", "r", encoding="utf-8") as stream:
        registry.update(yaml.safe_load(stream) or {})


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "hub_ok"}


@app.post("/run/{agent_name}")
async def run_agent(agent_name: str, request: Request) -> Dict[str, Any]:
    agents = registry.get("agents", {})
    if agent_name not in agents:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_name}")

    payload = await request.json()
    agent_cfg = agents[agent_name]
    url = f"{agent_cfg['url'].rstrip('/')}/run"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()
PY
fi

if [ "$SEED_HUB" = "true" ]; then
  seed_file "requirements.txt" $'fastapi\nuvicorn\nhttpx\npyyaml\n'

  seed_file "agent_registry.yaml" $'agents:\n  credit_appraisal:\n    url: "http://localhost:8091"\n    version: "1.0.0"\n  asset_appraisal:\n    url: "http://localhost:8092"\n    version: "1.0.0"\n'

  seed_file "README.md" $'# 🧠 AI Agent Hub\n\nThis repository orchestrates autonomous AI agents via a FastAPI gateway.\n\n## Development\n\n```bash\nuvicorn services.api.main:app --port 8090\n```\n\nUpdate `agent_registry.yaml` with each agent service endpoint.\n'
fi

log "Multi-repo bootstrap complete"
