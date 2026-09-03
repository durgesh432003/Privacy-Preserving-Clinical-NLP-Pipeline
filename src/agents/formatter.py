import json
import os
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from fhir.resources.bundle import Bundle
from pydantic import ValidationError
from src.agents.models import _clean_json
from src.agents.fhir_builder import build_fhir_bundle
from src.graph.state import AgentState

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "formatter.txt"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")


def formatter_node(state: AgentState) -> AgentState:
    extracted = state.get("extracted_data")
    grounded  = state.get("grounded_entities")
    assertion_map     = state.get("assertion_map")
    temporal_timeline = state.get("temporal_timeline")

    if not extracted:
        state["fhir_bundle"] = None
        state["fhir_valid"] = False
        return state

    model = state.get("model_name") or DEFAULT_MODEL
    from src.agents.model_registry import get_registry
    num_gpu = get_registry().get_num_gpu_for_model(model)
    llm = ChatOllama(model=model, temperature=0.0, base_url=OLLAMA_BASE_URL, num_gpu=num_gpu)

    payload_for_prompt = {
        "extracted_data": extracted,
        "grounded_ontologies": grounded or {}
    }
    prompt = Path(PROMPT_PATH).read_text().format(extracted_json=json.dumps(payload_for_prompt, indent=2))

    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        fhir_json = json.loads(_clean_json(raw))
        Bundle.model_validate(fhir_json)
        state["fhir_bundle"] = fhir_json
        state["fhir_valid"] = True
    except (json.JSONDecodeError, ValidationError, Exception):
        # Fall back to the deterministic builder — now enriched with assertion + temporal data
        fallback = build_fhir_bundle(
            extracted,
            grounded_entities=grounded,
            assertion_map=assertion_map,
            temporal_timeline=temporal_timeline,
        )
        try:
            Bundle.model_validate(fallback)
            state["fhir_valid"] = True
        except ValidationError:
            state["fhir_valid"] = False
        state["fhir_bundle"] = fallback

    return state
