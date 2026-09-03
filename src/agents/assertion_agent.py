"""
Assertion Agent Node for LangGraph pipeline.

Runs NegEx-style assertion/negation classification over all extracted clinical
entities and stores results in the pipeline state under `assertion_map`.

The assertion_map is consumed by:
  - formatter_node / fhir_builder.py  — to set Condition.verificationStatus
  - benchmark / evaluator             — to evaluate clinical accuracy

Assertion statuses:
  PRESENT      → active Condition (verificationStatus: confirmed)
  ABSENT       → excluded from FHIR Condition resources
  POSSIBLE     → Condition with verificationStatus: provisional
  HISTORICAL   → Condition with clinicalStatus: resolved
  FAMILY       → FamilyMemberHistory FHIR resource
  HYPOTHETICAL → excluded from FHIR Condition resources
"""

from src.graph.state import AgentState
from src.nlp.assertion_detector import classify_all_entities


def assertion_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node: classify assertion status for each extracted entity.

    Reads:  state["extracted_data"], state["input_text"]
    Writes: state["assertion_map"]
    """
    extracted = state.get("extracted_data")
    input_text = state.get("input_text", "")

    if not extracted or not input_text:
        state["assertion_map"] = {}
        return state

    try:
        assertion_map = classify_all_entities(input_text, extracted)
        state["assertion_map"] = assertion_map
    except Exception as e:
        # Graceful degradation: treat all entities as PRESENT
        state["assertion_map"] = {}

    return state
