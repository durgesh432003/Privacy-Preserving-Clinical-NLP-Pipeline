import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.agents.models import ValidationResult, _clean_json
from src.graph.state import AgentState

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "validator.txt"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")


def validate_icd_format(icd_codes: List[str]) -> List[str]:
    r"""Validates ICD-10 codes against the regex: ^[A-Z][0-9]{2}(\.\d{1,4})?$"""
    invalid_codes = []
    pattern = re.compile(r'^[A-Z][0-9]{2}(\.\d{1,4})?$')
    for code in icd_codes:
        if not pattern.match(code):
            invalid_codes.append(code)
    return invalid_codes


def check_field_completeness(data: Dict[str, Any]) -> List[str]:
    """Checks if mandatory fields are empty."""
    errors = []
    if not data.get("diagnoses"):
        errors.append("The 'diagnoses' list is empty. Are you sure there are no diagnoses in the text?")
    if not data.get("medications"):
        errors.append("The 'medications' list is empty. Please verify.")
    if not data.get("icd_codes"):
        errors.append("The 'icd_codes' list is empty. Please extract or infer ICD-10 codes if possible.")
    return errors


def validator_node(state: AgentState) -> AgentState:
    extracted = state.get("extracted_data")

    # If extraction failed entirely, mark invalid and let router retry
    if not extracted:
        state["validation_result"] = ValidationResult(
            is_valid=False, confidence=0.0, error_hints=state.get("error_hints", []), warnings=[]
        ).model_dump()
        return state

    error_hints = []

    # Rule-based checks
    bad_icds = validate_icd_format(extracted.get("icd_codes", []))
    if bad_icds:
        error_hints.append(f"Invalid ICD-10 format: {', '.join(bad_icds)}")
    error_hints.extend(check_field_completeness(extracted))

    # LLM plausibility check
    model = state.get("model_name") or DEFAULT_MODEL
    from src.agents.model_registry import get_registry
    num_gpu = get_registry().get_num_gpu_for_model(model)
    llm = ChatOllama(model=model, temperature=0.1, base_url=OLLAMA_BASE_URL, num_gpu=num_gpu)
    prompt = Path(PROMPT_PATH).read_text().format(
        input_text=state.get("input_text", ""),
        extracted_json=json.dumps(extracted, indent=2)
    )

    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        result = json.loads(_clean_json(raw))
        llm_valid = result.get("is_valid", True)
        confidence = float(result.get("confidence", 0.8))
        llm_hints = result.get("issues", [])
        llm_warnings = result.get("warnings", [])
    except Exception:
        llm_valid, confidence, llm_hints, llm_warnings = True, 0.7, [], []

    if not llm_valid:
        error_hints.extend(llm_hints)

    is_valid = len(error_hints) == 0
    state["validation_result"] = ValidationResult(
        is_valid=is_valid,
        confidence=confidence,
        error_hints=error_hints,
        warnings=llm_warnings,
    ).model_dump()
    return state
