import json
import re
import os
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.agents.models import ExtractedData, _clean_json
from src.graph.state import AgentState

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "extractor.txt"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")

def extractor_node(state: AgentState) -> AgentState:
    model = state.get("model_name") or DEFAULT_MODEL
    from src.agents.model_registry import get_registry
    num_gpu = get_registry().get_num_gpu_for_model(model)
    llm = ChatOllama(model=model, temperature=0.1, base_url=OLLAMA_BASE_URL, num_gpu=num_gpu)

    prompt = Path(PROMPT_PATH).read_text().format(
        input_text=state["input_text"],
        error_hints=(
            "\nPREVIOUS ERRORS TO FIX:\n"
            + "\n".join(f"- {h}" for h in state.get("error_hints", []))
            if state.get("error_hints")
            else ""
        ),
    )

    raw = None
    try:
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        data = json.loads(_clean_json(raw))
        ExtractedData(**data)  # validate shape
        state["extracted_data"] = data
        state["error_hints"] = []
    except Exception as e:
        if raw is None:
            state["extracted_data"] = None
            state["error_hints"] = [f"LLM failure: {str(e)}"]
            return state

        # Repair pass
        try:
            repair_prompt = (
                f"The following JSON is malformed. Fix it and return ONLY valid JSON:\n{raw}\nError: {e}"
            )
            raw2 = llm.invoke([HumanMessage(content=repair_prompt)]).content
            data = json.loads(_clean_json(raw2))
            ExtractedData(**data)
            state["extracted_data"] = data
            state["error_hints"] = []
        except Exception as e2:
            state["extracted_data"] = None
            state["error_hints"] = [str(e2)]

    return state
