import os
import logging
from dotenv import load_dotenv
load_dotenv()

# Suppress noisy Windows asyncio WinError 10054 (connection reset during teardown).
# This is a known CPython issue on Windows and does not affect functionality.
logging.getLogger("asyncio").addFilter(
    type("_Win10054Filter", (logging.Filter,), {
        "filter": lambda self, r: "WinError 10054" not in (r.getMessage() if r.exc_info is None
                                                           else str(r.exc_info))
    })()
)

import time
import httpx
import csv
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from src.graph.pipeline import create_pipeline
from src.agents.model_registry import get_registry
from src.agents.benchmark import run_benchmark as _run_benchmark, run_single_model
from src.agents.evaluator import evaluate_model_output
from src.evaluation.metrics import (
    EntityAnnotation,
    generate_full_report,
    pred_entities_from_state,
)

app = FastAPI(title="Clinical NLP Pipeline API", version="2.0.0")
pipeline = create_pipeline()

API_KEY = os.getenv("API_KEY", "local-dev-key-12345")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
RUNS_FILE = Path("data/runs.csv")
RUNS_FILE.parent.mkdir(exist_ok=True)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(key: str = Depends(api_key_header)):
    if key != API_KEY:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key


# ─── Request / Response Models ─────────────────────────────────────────────

class ProcessRequest(BaseModel):
    text: str
    model_name: str = ""
    max_retries: int = 3


class ProcessResponse(BaseModel):
    fhir_bundle: Optional[Dict[str, Any]]
    fhir_valid: bool
    extracted_data: Optional[Dict[str, Any]] = None
    grounded_entities: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    model_name: str = ""
    metrics: Dict[str, Any]
    # PhD Enhancement fields
    assertion_map: Optional[Dict[str, Any]] = None
    temporal_timeline: Optional[Dict[str, Any]] = None


class BenchmarkRequest(BaseModel):
    text: str
    model_names: List[str] = []
    max_retries: int = 3
    run_evaluation: bool = True


class ModelAddRequest(BaseModel):
    name: str
    display_name: str = ""
    category: str = "general"
    description: str = ""
    parameter_count: str = ""


class ModelToggleRequest(BaseModel):
    enabled: bool


# ─── Metrics Logging ───────────────────────────────────────────────────────

CSV_COLUMNS = [
    "timestamp", "model_name", "latency_ms", "retry_count", "confidence",
    "fhir_valid", "accuracy_score", "completeness_score", "fhir_score",
    "format_score", "overall_score",
]

def log_run(state: dict, evaluation: dict = None):
    """Log a pipeline run with per-model metrics to CSV."""
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_name": state.get("model_name", "unknown"),
        "latency_ms": round(state.get("latency_ms") or 0, 1),
        "retry_count": state.get("retry_count", 0),
        "confidence": state.get("confidence", 0.0),
        "fhir_valid": state.get("fhir_valid", False),
        "accuracy_score": "",
        "completeness_score": "",
        "fhir_score": "",
        "format_score": "",
        "overall_score": "",
    }
    if evaluation:
        row["accuracy_score"] = evaluation.get("accuracy", "")
        row["completeness_score"] = evaluation.get("completeness", "")
        row["fhir_score"] = evaluation.get("fhir_compliance", "")
        row["format_score"] = evaluation.get("format_quality", "")
        row["overall_score"] = evaluation.get("overall_score", "")

    write_header = not RUNS_FILE.exists() or RUNS_FILE.stat().st_size == 0
    with open(RUNS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── Core Endpoints ───────────────────────────────────────────────────────

@app.post("/process", response_model=ProcessResponse, dependencies=[Depends(get_api_key)])
async def process_record(request: ProcessRequest, background_tasks: BackgroundTasks):
    registry = get_registry()
    model = request.model_name or registry.get_default_model()

    state = pipeline.invoke({
        "input_text": request.text,
        "model_name": model,
        "max_retries": request.max_retries,
        "error_hints": [],
    })
    background_tasks.add_task(log_run, state)
    return ProcessResponse(
        fhir_bundle=state.get("fhir_bundle"),
        fhir_valid=state.get("fhir_valid", False),
        extracted_data=state.get("extracted_data"),
        grounded_entities=state.get("grounded_entities"),
        validation_result=state.get("validation_result"),
        model_name=model,
        metrics={
            "latency_ms": round(state.get("latency_ms") or 0, 1),
            "retry_count": state.get("retry_count", 0),
            "confidence": state.get("confidence", 0.0),
        },
        assertion_map=state.get("assertion_map"),
        temporal_timeline=state.get("temporal_timeline"),
    )


@app.post("/batch", response_model=List[ProcessResponse], dependencies=[Depends(get_api_key)])
async def batch_process(requests: List[ProcessRequest], background_tasks: BackgroundTasks):
    return [await process_record(r, background_tasks) for r in requests]


# ─── Benchmark Endpoint ───────────────────────────────────────────────────

@app.post("/benchmark", dependencies=[Depends(get_api_key)])
async def benchmark_endpoint(request: BenchmarkRequest, background_tasks: BackgroundTasks):
    registry = get_registry()
    model_names = request.model_names or registry.get_model_names()

    result = _run_benchmark(
        input_text=request.text,
        model_names=model_names,
        max_retries=request.max_retries,
        run_evaluation=request.run_evaluation,
    )

    # Log each model's run
    for mr in result.model_results:
        state_dict = {
            "model_name": mr.model_name,
            "latency_ms": mr.latency_ms,
            "retry_count": mr.retry_count,
            "confidence": mr.confidence,
            "fhir_valid": mr.fhir_valid,
        }
        background_tasks.add_task(log_run, state_dict, mr.evaluation)

    return result.to_dict()


# ─── Model Management Endpoints ───────────────────────────────────────────

@app.get("/models")
async def list_models():
    """List all available models from Ollama + config."""
    registry = get_registry()
    models = registry.get_available_models(include_disabled=True)
    return {
        "models": [m.to_dict() for m in models],
        "default": registry.get_default_model(),
        "evaluator": registry.get_evaluator_model(),
    }


@app.post("/models/add", dependencies=[Depends(get_api_key)])
async def add_model(request: ModelAddRequest):
    """Register a new model (must already be installed in Ollama)."""
    registry = get_registry()
    success = registry.add_model(
        name=request.name,
        display_name=request.display_name,
        category=request.category,
        description=request.description,
        parameter_count=request.parameter_count,
    )
    if success:
        return {"status": "ok", "message": f"Model '{request.name}' registered successfully"}
    return {"status": "error", "message": f"Model '{request.name}' not found in Ollama. Run: ollama pull {request.name}"}


@app.post("/models/{name}/toggle", dependencies=[Depends(get_api_key)])
async def toggle_model(name: str, request: ModelToggleRequest):
    """Enable or disable a model."""
    registry = get_registry()
    registry.toggle_model(name, request.enabled)
    return {"status": "ok", "model": name, "enabled": request.enabled}


# ─── Leaderboard & Metrics ────────────────────────────────────────────────

@app.get("/leaderboard")
async def get_leaderboard():
    """Get aggregated model performance rankings from historical runs."""
    if not RUNS_FILE.exists():
        return {"leaderboard": [], "total_runs": 0}

    import pandas as pd
    from io import StringIO

    try:
        df = pd.read_csv(RUNS_FILE)
    except Exception:
        return {"leaderboard": [], "total_runs": 0}

    if "model_name" not in df.columns or df.empty:
        return {"leaderboard": [], "total_runs": len(df)}

    # Filter out legacy rows without model_name
    df = df[df["model_name"].notna() & (df["model_name"] != "")]

    if df.empty:
        return {"leaderboard": [], "total_runs": 0}

    leaderboard = []
    for model_name, group in df.groupby("model_name"):
        entry = {
            "model_name": model_name,
            "total_runs": len(group),
            "avg_latency_ms": round(group["latency_ms"].astype(float).mean(), 1),
            "avg_confidence": round(group["confidence"].astype(float).mean(), 3),
            "fhir_success_rate": round(
                group["fhir_valid"].apply(lambda x: 1 if str(x).lower() == "true" else 0).mean() * 100, 1
            ),
        }

        # Add evaluation scores if available
        for col, key in [
            ("accuracy_score", "avg_accuracy"),
            ("completeness_score", "avg_completeness"),
            ("fhir_score", "avg_fhir_score"),
            ("format_score", "avg_format_score"),
            ("overall_score", "avg_overall_score"),
        ]:
            if col in group.columns:
                valid = pd.to_numeric(group[col], errors="coerce").dropna()
                entry[key] = round(valid.mean(), 1) if len(valid) > 0 else None
            else:
                entry[key] = None

        leaderboard.append(entry)

    # Sort by overall score descending, then by avg_confidence
    leaderboard.sort(
        key=lambda x: (x.get("avg_overall_score") or 0, x.get("avg_confidence") or 0),
        reverse=True,
    )

    return {"leaderboard": leaderboard, "total_runs": len(df)}


@app.get("/model/{name}/history")
async def model_history(name: str):
    """Get historical run data for a specific model."""
    if not RUNS_FILE.exists():
        return {"model": name, "runs": []}

    import pandas as pd

    try:
        df = pd.read_csv(RUNS_FILE)
    except Exception:
        return {"model": name, "runs": []}

    if "model_name" not in df.columns:
        return {"model": name, "runs": []}

    model_df = df[df["model_name"] == name]
    return {"model": name, "runs": model_df.to_dict(orient="records"), "total": len(model_df)}


# ─── Standardized F1 Evaluation Endpoint ─────────────────────────────────

class F1EvaluateRequest(BaseModel):
    """Run pipeline + compute standardized F1 metrics against a gold-standard fixture."""
    text: str
    model_name: str = ""
    max_retries: int = 3
    gold_entities: Optional[List[Dict[str, Any]]] = None  # override gold if provided
    gold_case_id: str = "SAMPLE-001"  # default to first sample case


@app.post("/evaluate/f1", dependencies=[Depends(get_api_key)])
async def evaluate_f1_endpoint(request: F1EvaluateRequest):
    """
    PhD-grade evaluation endpoint.

    Runs the full pipeline on `text`, then computes publication-grade metrics:
    - Strict/Soft Entity F1 (per category + macro average)
    - Ontology Grounding Top-1/Top-3 Accuracy
    - FHIR Entry Validity Rate
    - Assertion Classification Accuracy
    - Temporal Association Accuracy

    If `gold_entities` is not provided in the request body, the server loads
    the gold annotation from `data/eval/sample_gold.json` using `gold_case_id`.
    """
    import json as _json
    from pathlib import Path as _Path

    registry = get_registry()
    model = request.model_name or registry.get_default_model()

    # 1. Run pipeline
    state = pipeline.invoke({
        "input_text": request.text,
        "model_name": model,
        "max_retries": request.max_retries,
        "error_hints": [],
    })

    # 2. Load gold entities
    gold_annotation_list: List[Dict[str, Any]] = []
    if request.gold_entities:
        gold_annotation_list = request.gold_entities
    else:
        gold_file = _Path("data/eval/sample_gold.json")
        if gold_file.exists():
            try:
                gold_data = _json.loads(gold_file.read_text())
                for case in gold_data.get("cases", []):
                    if case.get("case_id") == request.gold_case_id:
                        gold_annotation_list = case.get("gold_entities", [])
                        break
            except Exception:
                pass

    gold_entities = [
        EntityAnnotation(
            text=e["text"],
            category=e["category"],
            code=e.get("code", ""),
            code_system=e.get("code_system", ""),
            assertion=e.get("assertion", "PRESENT"),
            onset_date=e.get("onset_date"),
        )
        for e in gold_annotation_list
    ]

    # 3. Build predicted entities from state
    pred_entities = pred_entities_from_state(state)

    # 4. Generate full evaluation report
    report = generate_full_report(
        gold_entities=gold_entities,
        pred_entities=pred_entities,
        fhir_bundle=state.get("fhir_bundle"),
        pred_assertion_map=state.get("assertion_map"),
        pred_timeline=state.get("temporal_timeline"),
        model_name=model,
        case_id=request.gold_case_id,
    )

    return {
        "model_name": model,
        "case_id": request.gold_case_id,
        "report": report.to_dict(),
        "pipeline_metrics": {
            "latency_ms": round(state.get("latency_ms") or 0, 1),
            "retry_count": state.get("retry_count", 0),
            "fhir_valid": state.get("fhir_valid", False),
        },
    }


# ─── Health & Export ───────────────────────────────────────────────────────

@app.get("/health")
async def health():
    status = {"status": "ok", "ollama": False, "models_available": 0}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
            if r.status_code == 200:
                status["ollama"] = True
                models = r.json().get("models", [])
                status["models_available"] = len(models)
                status["model_names"] = [m.get("name", "") for m in models]
    except Exception:
        pass
    return status


@app.get("/metrics/export", dependencies=[Depends(get_api_key)])
async def export_metrics():
    if not RUNS_FILE.exists():
        return {"error": "No metrics yet"}
    return FileResponse(path=RUNS_FILE, filename="runs.csv", media_type="text/csv")
