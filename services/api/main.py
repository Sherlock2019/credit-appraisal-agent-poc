# ~/demo-library/services/api/main.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.routers import agents, reports, settings
from services.api.routers import training  # <-- added

app = FastAPI(title="Credit Appraisal API")

# CORS for local dev (tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/health")
def health():
    return {"ok": True}

app.include_router(agents.router)
app.include_router(reports.router)
app.include_router(settings.router)
app.include_router(training.router)  # <-- added
