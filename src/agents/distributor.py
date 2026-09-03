import json
import os
import re
import uuid
from pathlib import Path
from src.graph.state import AgentState

def sanitize_folder_name(name: str) -> str:
    """Sanitize the disease name to make it a valid folder name."""
    # Replace non-alphanumeric characters (except spaces) with underscores
    name = re.sub(r'[^a-zA-Z0-9 ]', '_', name)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    return name.strip('_')

def distributor_node(state: AgentState) -> AgentState:
    """
    Distributes the final JSON output into local file storage,
    organized by disease folder and patient case ID.
    """
    # 1. Determine disease
    disease = "Unknown_Disease"
    extracted = state.get("extracted_data")
    if extracted and extracted.get("diagnoses"):
        first_diag = extracted["diagnoses"][0]
        if first_diag:
            disease = sanitize_folder_name(first_diag)

    # 2. Generate a patient case ID
    case_id = str(uuid.uuid4())

    # 3. Ensure destination directory exists
    base_dir = Path("extracted data")
    disease_dir = base_dir / disease
    disease_dir.mkdir(parents=True, exist_ok=True)

    # 4. Save JSON output
    file_path = disease_dir / f"{case_id}.json"
    
    # Save the FHIR bundle as the final output
    output_data = state.get("fhir_bundle")
    if not output_data:
        # Fallback to extracted data if fhir_bundle is missing for some reason
        output_data = state.get("extracted_data", {})

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # 5. Store saved path in state
    state["saved_path"] = str(file_path)

    return state
