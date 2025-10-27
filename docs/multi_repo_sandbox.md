# AI Agent Sandbox Integration Guide

This document explains how to plug the **Credit Appraisal Agent** into the global
[`AI-AIGENTbythePeoplesANDBOX`](https://github.com/Sherlock2019/AI-AIGENTbythePeoplesANDBOX.git)
scaffold while keeping every agent in its own repository.

The approach mirrors the "one agent = one repo" strategy so that the credit
appraisal service can be orchestrated alongside the asset appraisal agent and
any future micro-agents (KYC, AML, ESG, etc.).

## Repository Layout

```text
AI-AIGENTbythePeoplesANDBOX/
├── ai-agent-hub/                # FastAPI gateway + optional UI
│   ├── services/api/main.py
│   ├── configs/
│   └── agent_registry.yaml
├── agent-credit-appraisal/      # (this repo) as a git submodule
├── agent-asset-appraisal/       # https://github.com/Sherlock2019/asset-appraisal-agent-.git
├── shared-agent-sdk/            # shared utilities (data loaders, auth, logging)
└── ...                          # future agents (kyc, aml, fraud, ...)
```

Each agent remains an independent Python package and exposes the same REST
surface:

- `GET /health` – readiness probe
- `POST /run` – inference endpoint
- `POST /train` – optional fine-tuning hook

## Quick Start Script

The repository now ships with `scripts/init_multi_repo.sh`.  Execute it from any
workspace to bootstrap the directory structure above and optionally clone the
credit and asset appraisal services as git submodules inside the global
sandbox repo.

```bash
chmod +x scripts/init_multi_repo.sh
./scripts/init_multi_repo.sh --workspace ~/rackspace-aisandbox \
  --hub-ref main --credit-ref main --asset-ref main
```

### What the Script Does

1. Creates the umbrella folder (default: `~/rackspace-aisandbox`, override with
   `--workspace`).
2. Clones the orchestrator repo
   [`AI-AIGENTbythePeoplesANDBOX`](https://github.com/Sherlock2019/AI-AIGENTbythePeoplesANDBOX.git)
   and checks out the requested branch or tag (`--hub-ref`).
3. Adds the credit and asset appraisal repos as submodules beneath the
   orchestrator so that each agent keeps its own git history. You can pin a
   specific release with `--credit-ref` or `--asset-ref`.
4. Drops boilerplate FastAPI apps, requirements files, and the agent registry if
   the orchestrator repo was empty (useful for a first-time bootstrap). Supply
   `--skip-seed` to leave an existing hub untouched.

Run `./scripts/init_multi_repo.sh --help` to view all switches, including
`--shared-sdk-repo`/`--shared-sdk-ref` for wiring in additional shared
components.

> ℹ️  Re-run the script safely; it skips assets that already exist and will
> refresh git submodules with the latest `main` branch.

## Registering Agents with the Hub

> 💡 Need to add another shared package? Pass `--shared-sdk-repo` and
> `--shared-sdk-ref` so the bootstrapper wires in the additional module as a
> submodule alongside the agents.

1. Launch each agent locally:
   ```bash
   uvicorn agent.main:app --port 8091  # credit agent
   uvicorn agent.main:app --port 8092  # asset agent
   ```
2. Update `infra/agent_registry.yaml` (copy from the provided
   `agent_registry.example.yaml`) with the service URLs and versions.
3. Start the hub:
   ```bash
   uvicorn ai_agent_hub.services.api.main:app --port 8090
   ```
4. Test the routing:
   ```bash
   curl -X POST http://localhost:8090/run/credit_appraisal \
     -H "Content-Type: application/json" \
     -d '{"text": "customer income is stable"}'
   ```

## Data & Model Sharing

- Publish reusable artefacts (datasets, checkpoints, prompts) to S3, Hugging
  Face Hub, or an internal registry.
- Use the optional `shared-agent-sdk` repository for anonymisation utilities,
  storage clients, and telemetry code that can be installed via
  `pip install -e shared-agent-sdk/`.

## CI/CD Notes

- Submodule layout keeps each agent below the 50 MB threshold, making CI/CD and
  dependency caching straightforward.
- Deploy or retrain one agent without touching others—only the hub reads the
  registry to discover available capabilities.
- Add GitHub Actions in each agent repo for per-agent unit tests and automated
  Docker builds.

## Next Steps

- Add Dockerfiles and a `docker-compose.yml` in the orchestrator repo so the hub
  plus agents can be launched with a single command.
- Create a shared `agent-protocol` Python package (or reuse
  `shared-agent-sdk`) that standardises request/response schemas across agents.
- Instrument each agent with Prometheus exporters so the hub can surface
  health/usage metrics in Grafana.
