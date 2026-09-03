"""
Ontology Grounder Agent Node for LangGraph pipeline.
Receives extracted clinical entities and grounds them using the local semantic ontology engine.
Enriches entities with SNOMED CT, RxNorm, LOINC, and ICD-10-CM codes.
"""

from typing import Dict, Any
from src.graph.state import AgentState
from src.ontology.grounder import get_grounder


def ontology_grounder_node(state: AgentState) -> AgentState:
    """
    LangGraph agent node that grounds extracted clinical entities to standardized medical ontologies.
    Operates 100% locally and deterministically.
    """
    extracted_data = state.get("extracted_data")
    if not extracted_data:
        state["grounded_entities"] = None
        return state

    try:
        grounder = get_grounder()
        grounded = grounder.ground_extracted_data(extracted_data)
        state["grounded_entities"] = grounded
    except Exception as e:
        # Graceful fallback: set grounded_entities to empty structure
        state["grounded_entities"] = {
            "diagnoses": [],
            "medications": [],
            "labs": [],
            "procedures": [],
            "error": str(e)
        }

    return state
