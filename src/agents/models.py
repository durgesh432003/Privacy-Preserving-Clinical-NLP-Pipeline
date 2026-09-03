from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Medication(BaseModel):
    name: str = Field(description="Name of the medication")
    dose: Optional[str] = Field(None, description="Dosage of the medication (e.g., '10 mg')")
    frequency: Optional[str] = Field(None, description="Frequency of administration (e.g., 'daily', 'BID')")

class ExtractedData(BaseModel):
    diagnoses: List[str] = Field(default_factory=list, description="List of medical diagnoses")
    medications: List[Medication] = Field(default_factory=list, description="List of medications prescribed")
    procedures: List[str] = Field(default_factory=list, description="List of medical procedures performed")
    icd_codes: List[str] = Field(default_factory=list, description="List of ICD-10 codes mentioned")
    dates: Dict[str, str] = Field(default_factory=dict, description="Key dates extracted (e.g., admission, discharge)")

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the extracted data is valid according to the rules")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    error_hints: List[str] = Field(default_factory=list, description="List of errors or hints for correction")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings about the data")

def _clean_json(text: str) -> str:
    """Strip markdown fences and find the first JSON object."""
    import re
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text
