import time
from langgraph.graph import StateGraph, END
from src.graph.state import AgentState
from src.agents.extractor import extractor_node
from src.agents.validator import validator_node
from src.agents.ontology_agent import ontology_grounder_node
from src.agents.formatter import formatter_node
from src.agents.distributor import distributor_node
from src.agents.assertion_agent import assertion_agent_node
from src.agents.temporal_agent import temporal_agent_node

def start_tracking(state: AgentState) -> AgentState:
    """Initializes tracking variables if not present."""
    if "start_time" not in state or state["start_time"] == 0:
        state["start_time"] = time.time()
    
    # Increment retry count
    if "retry_count" not in state:
        state["retry_count"] = 0
    else:
        state["retry_count"] += 1
        
    return state

def end_tracking(state: AgentState) -> AgentState:
    """Calculates latency at the end of the pipeline."""
    if "start_time" in state and state["start_time"] > 0:
        state["latency_ms"] = (time.time() - state["start_time"]) * 1000.0
    else:
        state["latency_ms"] = 0.0
    
    # Pull confidence out of validation_result if not already set
    if not state.get("confidence"):
        val = state.get("validation_result") or {}
        state["confidence"] = val.get("confidence", 0.0)
    
    return state

def should_retry(state: AgentState) -> str:
    """Routes to extractor if invalid and retries remain, else ontology_grounder."""
    validation = state.get("validation_result", {})
    if isinstance(validation, str):
        import json
        validation = json.loads(validation)
        
    is_valid = validation.get("is_valid", False)
    retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    
    if not is_valid and retries < max_retries:
        return "extractor"
    return "assertion_agent"

def create_pipeline():
    """Creates and compiles the LangGraph StateGraph pipeline."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("start_tracking", start_tracking)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("assertion_agent", assertion_agent_node)   # NEW: negation detection
    workflow.add_node("ontology_grounder", ontology_grounder_node)
    workflow.add_node("temporal_agent", temporal_agent_node)     # NEW: temporal timeline
    workflow.add_node("formatter", formatter_node)
    workflow.add_node("distributor", distributor_node)
    workflow.add_node("end_tracking", end_tracking)
    
    # Define edges
    workflow.set_entry_point("start_tracking")
    workflow.add_edge("start_tracking", "extractor")
    workflow.add_edge("extractor", "validator")
    
    # Conditional edge after validator:
    # - If invalid AND retries remain → retry extraction
    # - If valid OR max retries hit → run assertion detection
    workflow.add_conditional_edges(
        "validator",
        should_retry,
        {
            "assertion_agent": "assertion_agent",     # proceed to PhD path
            "extractor": "start_tracking"             # loop back and increment retry count
        }
    )
    
    # After assertion detection → ontology grounding
    workflow.add_edge("assertion_agent", "ontology_grounder")
    # After grounding → temporal timeline extraction
    workflow.add_edge("ontology_grounder", "temporal_agent")
    # Temporal → formatter (now has assertion_map + temporal_timeline in state)
    workflow.add_edge("temporal_agent", "formatter")
    workflow.add_edge("formatter", "distributor")
    workflow.add_edge("distributor", "end_tracking")
    workflow.add_edge("end_tracking", END)
    
    return workflow.compile()
