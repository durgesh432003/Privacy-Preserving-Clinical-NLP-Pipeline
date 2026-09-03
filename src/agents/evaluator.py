"""
LLM-as-Judge Evaluator — Uses the strongest available model to score
other models' extraction outputs on accuracy, completeness, FHIR compliance,
and format quality.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.agents.models import _clean_json

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evaluator.txt"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Scoring weights
WEIGHTS = {
    "accuracy": 0.40,
    "completeness": 0.30,
    "fhir_compliance": 0.20,
    "format_quality": 0.10,
}


@dataclass
class EvaluationResult:
    """Structured evaluation scores from the judge model."""
    model_name: str
    accuracy: float = 0.0
    completeness: float = 0.0
    fhir_compliance: float = 0.0
    format_quality: float = 0.0
    overall_score: float = 0.0
    reasoning: Dict[str, str] = field(default_factory=dict)
    hallucinations_found: list = field(default_factory=list)
    missed_entities: list = field(default_factory=list)
    evaluator_model: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_model_output(
    input_text: str,
    extracted_data: Optional[Dict[str, Any]],
    fhir_bundle: Optional[Dict[str, Any]],
    fhir_valid: bool,
    model_name: str,
    evaluator_model: str = "",
) -> EvaluationResult:
    """
    Evaluate a model's pipeline output using an LLM judge.

    Args:
        input_text: Original discharge summary text
        extracted_data: The model's extracted JSON output
        fhir_bundle: The generated FHIR bundle
        fhir_valid: Whether the FHIR bundle passed validation
        model_name: Name of the model being evaluated
        evaluator_model: Name of the judge model to use

    Returns:
        EvaluationResult with scores and reasoning
    """
    if not evaluator_model:
        from src.agents.model_registry import get_registry
        evaluator_model = get_registry().get_evaluator_model()

    result = EvaluationResult(
        model_name=model_name,
        evaluator_model=evaluator_model,
    )

    # If extraction completely failed, give zeros
    if not extracted_data:
        result.error = "Extraction failed completely — no data to evaluate"
        return result

    # Build the evaluation prompt
    prompt_template = Path(PROMPT_PATH).read_text()
    prompt = prompt_template.format(
        input_text=input_text,
        extracted_json=json.dumps(extracted_data, indent=2),
        fhir_json=json.dumps(fhir_bundle, indent=2) if fhir_bundle else "null",
        fhir_valid=str(fhir_valid),
    )

    from src.agents.model_registry import get_registry
    num_gpu = get_registry().get_num_gpu_for_model(evaluator_model)
    llm = ChatOllama(
        model=evaluator_model,
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_gpu=num_gpu,
    )

    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        scores = json.loads(_clean_json(raw))

        result.accuracy = float(scores.get("accuracy", 0))
        result.completeness = float(scores.get("completeness", 0))
        result.fhir_compliance = float(scores.get("fhir_compliance", 0))
        result.format_quality = float(scores.get("format_quality", 0))
        result.reasoning = scores.get("reasoning", {})
        result.hallucinations_found = scores.get("hallucinations_found", [])
        result.missed_entities = scores.get("missed_entities", [])

        # Calculate weighted overall score
        result.overall_score = round(
            result.accuracy * WEIGHTS["accuracy"]
            + result.completeness * WEIGHTS["completeness"]
            + result.fhir_compliance * WEIGHTS["fhir_compliance"]
            + result.format_quality * WEIGHTS["format_quality"],
            2,
        )

    except Exception as e:
        result.error = f"Evaluation failed: {str(e)}"
        # Provide basic heuristic scores as fallback
        if extracted_data:
            result.format_quality = 60.0  # JSON was parseable
            result.accuracy = 50.0  # Unknown
            result.completeness = 50.0  # Unknown
            result.fhir_compliance = 80.0 if fhir_valid else 20.0
            result.overall_score = round(
                result.accuracy * WEIGHTS["accuracy"]
                + result.completeness * WEIGHTS["completeness"]
                + result.fhir_compliance * WEIGHTS["fhir_compliance"]
                + result.format_quality * WEIGHTS["format_quality"],
                2,
            )

    return result
