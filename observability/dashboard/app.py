"""
FastAPI backend for the XSAKE real-time training dashboard.

Endpoints:
  GET /metrics         — current training metrics (loss, ppl, grad_norm, tokens)
  GET /benchmark       — benchmark results (latency, memory, throughput)
  GET /hads            — HADS profile (per-head sparsity and entropy)
  GET /health          — liveness probe

The frontend (dashboard/frontend/dashboard.html) polls these endpoints
and renders live Plotly.js charts.

Start:
    uvicorn observability.dashboard.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from observability.dashboard.metrics_store import MetricsStore


app = FastAPI(title="XSAKE Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

store = MetricsStore()

RESULTS_DIR = Path("benchmarks/results")
FRONTEND    = Path("observability/dashboard/frontend/dashboard.html")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    if FRONTEND.exists():
        return FRONTEND.read_text()
    return "<h1>XSAKE Dashboard — frontend not found</h1>"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    """Return training metrics history for live plotting."""
    return JSONResponse(store.get_metrics())


@app.post("/metrics")
async def push_metrics(data: dict):
    """Training loop posts metrics here at each log step."""
    store.push(data)
    return {"ok": True}


@app.get("/benchmark")
async def benchmark():
    """Return pre-computed benchmark results for the dashboard."""
    results = {}
    for name in ["latency_results", "memory_results", "throughput_results", "hads_results"]:
        path = RESULTS_DIR / f"{name}.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return JSONResponse(results)


@app.get("/hads")
async def hads_profile():
    """Return the current HADS sparsity profile."""
    path = RESULTS_DIR / "hads_results.json"
    if path.exists():
        with open(path) as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"error": "No HADS results found. Run benchmarks/hads_benchmark.py first."})


@app.get("/benchmark/latency")
async def latency():
    path = RESULTS_DIR / "latency_results.json"
    if path.exists():
        with open(path) as f:
            return JSONResponse(json.load(f))
    return JSONResponse({"error": "Run benchmarks/latency_benchmark.py first."})
