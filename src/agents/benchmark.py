"""
Benchmark Runner — Runs the same input through multiple models and collects
comparable metrics for side-by-side evaluation.
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from src.graph.pipeline import create_pipeline
from src.agents.evaluator import evaluate_model_output, EvaluationResult
from src.agents.model_registry import get_registry


@dataclass
class ModelRunResult:
    """Result of running a single model through the pipeline."""
    model_name: str
    extracted_data: Optional[Dict[str, Any]] = None
    validation_result: Optional[Dict[str, Any]] = None
    fhir_bundle: Optional[Dict[str, Any]] = None
    fhir_valid: bool = False
    latency_ms: float = 0.0
    retry_count: int = 0
    confidence: float = 0.0
    evaluation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Aggregated result of benchmarking multiple models."""
    input_text: str
    model_results: List[ModelRunResult] = field(default_factory=list)
    evaluator_model: str = ""
    total_time_ms: float = 0.0
    winner: str = ""
    ranking: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Truncate input text for response size
        d["input_text"] = d["input_text"][:200] + "..." if len(d["input_text"]) > 200 else d["input_text"]
        return d


def run_single_model(input_text: str, model_name: str, max_retries: int = 3) -> ModelRunResult:
    """Run the pipeline with a specific model and return results."""
    result = ModelRunResult(model_name=model_name)

    try:
        pipeline = create_pipeline()
        state = pipeline.invoke({
            "input_text": input_text,
            "model_name": model_name,
            "max_retries": max_retries,
            "error_hints": [],
        })

        result.extracted_data = state.get("extracted_data")
        result.validation_result = state.get("validation_result")
        result.fhir_bundle = state.get("fhir_bundle")
        result.fhir_valid = state.get("fhir_valid", False)
        result.latency_ms = round(state.get("latency_ms", 0), 1)
        result.retry_count = state.get("retry_count", 0)
        result.confidence = state.get("confidence", 0.0)

    except Exception as e:
        result.error = str(e)

    return result


def run_benchmark(
    input_text: str,
    model_names: List[str],
    max_retries: int = 3,
    run_evaluation: bool = True,
    evaluator_model: str = "",
) -> BenchmarkResult:
    """
    Run the same input through multiple models and compare results.

    Args:
        input_text: The discharge summary text to process
        model_names: List of model names to benchmark
        max_retries: Max retries per model
        run_evaluation: Whether to run the LLM-as-judge evaluator
        evaluator_model: Override for the judge model

    Returns:
        BenchmarkResult with all model outputs, evaluations, and rankings
    """
    if not evaluator_model:
        evaluator_model = get_registry().get_evaluator_model()

    benchmark = BenchmarkResult(
        input_text=input_text,
        evaluator_model=evaluator_model,
    )

    total_start = time.time()

    # Run each model sequentially (parallel would require more VRAM)
    for model_name in model_names:
        model_result = run_single_model(input_text, model_name, max_retries)

        # Run evaluation if requested
        if run_evaluation and model_result.extracted_data:
            eval_result = evaluate_model_output(
                input_text=input_text,
                extracted_data=model_result.extracted_data,
                fhir_bundle=model_result.fhir_bundle,
                fhir_valid=model_result.fhir_valid,
                model_name=model_name,
                evaluator_model=evaluator_model,
            )
            model_result.evaluation = eval_result.to_dict()

        benchmark.model_results.append(model_result)

    benchmark.total_time_ms = round((time.time() - total_start) * 1000, 1)

    # Compute rankings
    ranked = []
    for mr in benchmark.model_results:
        overall = 0.0
        if mr.evaluation:
            overall = mr.evaluation.get("overall_score", 0.0)
        ranked.append({
            "model_name": mr.model_name,
            "overall_score": overall,
            "latency_ms": mr.latency_ms,
            "fhir_valid": mr.fhir_valid,
            "confidence": mr.confidence,
        })

    ranked.sort(key=lambda x: x["overall_score"], reverse=True)
    benchmark.ranking = ranked

    if ranked:
        benchmark.winner = ranked[0]["model_name"]

    return benchmark
