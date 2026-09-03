from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict):
    input_text: str
    model_name: str  # Ollama model to use for this run
    extracted_data: Optional[Dict[str, Any]]
    grounded_entities: Optional[Dict[str, Any]]
    validation_result: Optional[Dict[str, Any]]
    error_hints: List[str]
    retry_count: int
    max_retries: int
    fhir_bundle: Optional[Dict[str, Any]]
    fhir_valid: bool
    confidence: float
    latency_ms: float
    start_time: float
    saved_path: Optional[str]

    # Assertion & negation map — populated by assertion_agent_node.
    # Structure: { "diagnoses": { "Chest pain": "ABSENT", ... },
    #              "medications": { ... }, "procedures": { ... } }
    assertion_map: Optional[Dict[str, Dict[str, str]]]

    # Temporal timeline — populated by temporal_agent_node.
    # Structure: { "expressions": [...], "entity_times": { "diagnoses": {...}, ... } }
    temporal_timeline: Optional[Dict[str, Any]]
