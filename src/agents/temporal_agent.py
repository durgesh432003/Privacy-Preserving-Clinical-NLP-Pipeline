"""
Temporal Agent Node for LangGraph pipeline.

Extracts temporal expressions from the input clinical text and associates
them with the extracted clinical entities. Results are stored in the pipeline
state under `temporal_timeline`.

The temporal_timeline is consumed by:
  - fhir_builder.py — to populate onsetDateTime, occurrenceDateTime, etc.
  - dashboard UI    — to show a chronological event timeline

Output structure in state["temporal_timeline"]:
{
  "expressions": [ { raw_text, normalized_date, temporal_type, ... }, ... ],
  "entity_times": {
    "diagnoses":   { "Pneumonia": "2024-01-10", ... },
    "procedures":  { "Chest X-ray": "on admission", ... },
    "medications": { "Aspirin": null, ... }
  }
}
"""

from src.graph.state import AgentState
from src.nlp.temporal_extractor import build_temporal_timeline


def temporal_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: extract and associate temporal expressions with entities.

    Reads:  state["input_text"], state["extracted_data"]
    Writes: state["temporal_timeline"]
    """
    input_text = state.get("input_text", "")
    extracted = state.get("extracted_data")

    if not input_text or not extracted:
        state["temporal_timeline"] = {}
        return state

    try:
        timeline = build_temporal_timeline(input_text, extracted)
        state["temporal_timeline"] = timeline
    except Exception as e:
        state["temporal_timeline"] = {"error": str(e), "expressions": [], "entity_times": {}}

    return state
