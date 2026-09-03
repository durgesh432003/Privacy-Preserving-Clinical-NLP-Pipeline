import uuid
from typing import Dict, Any, Optional
from src.ontology.grounder import get_grounder
from src.ontology.ontologies import SYSTEM_SNOMED, SYSTEM_RXNORM, SYSTEM_LOINC, SYSTEM_ICD10

# Assertion status constants
_ABSENT      = "ABSENT"
_POSSIBLE    = "POSSIBLE"
_HISTORICAL  = "HISTORICAL"
_HYPOTHETICAL = "HYPOTHETICAL"
_FAMILY      = "FAMILY"
_PRESENT     = "PRESENT"


def _get_assertion_status(entity: str, category: str, assertion_map: Optional[Dict]) -> str:
    """Look up assertion status for an entity, defaulting to PRESENT."""
    if not assertion_map:
        return _PRESENT
    return assertion_map.get(category, {}).get(entity, _PRESENT)


def _get_entity_datetime(entity: str, category: str, temporal_timeline: Optional[Dict]) -> Optional[str]:
    """Look up the associated normalized date for an entity from the temporal timeline."""
    if not temporal_timeline:
        return None
    entity_times = temporal_timeline.get("entity_times", {})
    return entity_times.get(category, {}).get(entity)


def _condition_clinical_status(assertion_status: str) -> dict:
    """Map assertion status to FHIR Condition.clinicalStatus coding."""
    if assertion_status == _HISTORICAL:
        return {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": "resolved",
                "display": "Resolved"
            }]
        }
    return {
        "coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
            "code": "active",
            "display": "Active"
        }]
    }


def _condition_verification_status(assertion_status: str) -> dict:
    """Map assertion status to FHIR Condition.verificationStatus coding."""
    if assertion_status in (_ABSENT, _HYPOTHETICAL):
        return {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "code": "refuted",
                "display": "Refuted"
            }]
        }
    if assertion_status == _POSSIBLE:
        return {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "code": "provisional",
                "display": "Provisional"
            }]
        }
    if assertion_status == _FAMILY:
        return {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "code": "differential",
                "display": "Differential (Family History)"
            }]
        }
    return {
        "coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
            "code": "confirmed",
            "display": "Confirmed"
        }]
    }


def build_fhir_bundle(
    extracted_data: Dict[str, Any],
    grounded_entities: Optional[Dict[str, Any]] = None,
    assertion_map: Optional[Dict[str, Dict[str, str]]] = None,
    temporal_timeline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministically builds a fully grounded FHIR R4 Bundle from ExtractedData,
    GroundedConcepts, assertion status, and temporal timeline.

    PhD Enhancements:
    - Respects assertion/negation status per entity:
        ABSENT/HYPOTHETICAL → verificationStatus: refuted (still included for audit trail)
        POSSIBLE            → verificationStatus: provisional
        HISTORICAL          → clinicalStatus: resolved
        FAMILY              → included with family-history note
        PRESENT             → active, confirmed (default)
    - Populates onsetDateTime on Condition from temporal timeline
    - Populates occurrenceDateTime on Procedure from temporal timeline
    """
    if not grounded_entities:
        try:
            grounder = get_grounder()
            grounded_entities = grounder.ground_extracted_data(extracted_data)
        except Exception:
            grounded_entities = {"diagnoses": [], "medications": [], "labs": [], "procedures": []}

    entries = []

    # --- 1. Diagnoses → Condition (with SNOMED CT + ICD-10 codings) ---
    grounded_diagnoses = grounded_entities.get("diagnoses", [])
    raw_diagnoses = extracted_data.get("diagnoses", [])

    for idx, diag_text in enumerate(raw_diagnoses):
        g_info = grounded_diagnoses[idx] if idx < len(grounded_diagnoses) else {}
        codings = []

        system  = g_info.get("system") or SYSTEM_SNOMED
        code    = g_info.get("code") or "UNMAPPED"
        display = g_info.get("display") or diag_text

        codings.append({"system": system, "code": code, "display": display})

        # Dual-coding: add ICD-10 if grounded
        if g_info.get("icd10_code"):
            codings.append({
                "system": SYSTEM_ICD10,
                "code": g_info["icd10_code"],
                "display": display,
            })

        # Assertion status
        assertion_status = _get_assertion_status(diag_text, "diagnoses", assertion_map)
        clinical_status  = _condition_clinical_status(assertion_status)
        verification_status = _condition_verification_status(assertion_status)

        condition: Dict[str, Any] = {
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": clinical_status,
                "verificationStatus": verification_status,
                "code": {
                    "coding": codings,
                    "text": diag_text
                },
                "subject": {"reference": "Patient/1"},
            }
        }

        # Temporal: onset date
        onset_dt = _get_entity_datetime(diag_text, "diagnoses", temporal_timeline)
        if onset_dt and not onset_dt.startswith("on ") and not onset_dt.startswith("at "):
            # ISO date string
            condition["resource"]["onsetDateTime"] = onset_dt
        elif onset_dt:
            # Named anchor — store as note
            condition["resource"]["note"] = [{"text": f"Temporal context: {onset_dt}"}]

        # Add assertion status as a FHIR extension for traceability
        if assertion_status != _PRESENT:
            condition["resource"]["extension"] = [{
                "url": "http://example.org/fhir/StructureDefinition/assertion-status",
                "valueString": assertion_status
            }]

        entries.append(condition)

    # --- 2. Raw ICD Codes → Condition resources ---
    for icd in extracted_data.get("icd_codes", []):
        condition = {
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Condition",
                "clinicalStatus": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                        "display": "Active"
                    }]
                },
                "verificationStatus": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "confirmed",
                        "display": "Confirmed"
                    }]
                },
                "code": {
                    "coding": [{"system": SYSTEM_ICD10, "code": icd}]
                },
                "subject": {"reference": "Patient/1"}
            }
        }
        entries.append(condition)

    # --- 3. Medications → MedicationStatement (with RxNorm codings) ---
    grounded_meds = grounded_entities.get("medications", [])
    raw_meds = extracted_data.get("medications", [])

    for idx, med in enumerate(raw_meds):
        name = med.get("name", "Unknown") if isinstance(med, dict) else str(med)
        dose_parts = [med.get("dose") or "", med.get("frequency") or ""] if isinstance(med, dict) else []
        dosage_text = " ".join(p for p in dose_parts if p).strip()

        g_info  = grounded_meds[idx] if idx < len(grounded_meds) else {}
        system  = g_info.get("system") or SYSTEM_RXNORM
        code    = g_info.get("code") or "UNMAPPED"
        display = g_info.get("display") or name

        med_statement: Dict[str, Any] = {
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "MedicationStatement",
                "status": "active",
                "medication": {
                    "concept": {
                        "coding": [{"system": system, "code": code, "display": display}],
                        "text": name
                    }
                },
                "subject": {"reference": "Patient/1"}
            }
        }
        if dosage_text:
            med_statement["resource"]["dosage"] = [{"text": dosage_text}]

        entries.append(med_statement)

    # --- 4. Labs → Observation (with LOINC codings) ---
    grounded_labs = grounded_entities.get("labs", [])
    raw_labs = extracted_data.get("labs", []) or []

    for idx, lab in enumerate(raw_labs):
        name = lab.get("name", "Unknown") if isinstance(lab, dict) else str(lab)
        val  = lab.get("value") if isinstance(lab, dict) else None
        unit = lab.get("unit") if isinstance(lab, dict) else None

        g_info  = grounded_labs[idx] if idx < len(grounded_labs) else {}
        system  = g_info.get("system") or SYSTEM_LOINC
        code    = g_info.get("code") or "UNMAPPED"
        display = g_info.get("display") or name

        observation: Dict[str, Any] = {
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [{"system": system, "code": code, "display": display}],
                    "text": name
                },
                "subject": {"reference": "Patient/1"}
            }
        }
        if val is not None:
            try:
                observation["resource"]["valueQuantity"] = {
                    "value": float(val),
                    "unit": unit or "",
                    "system": "http://unitsofmeasure.org"
                }
            except (ValueError, TypeError):
                observation["resource"]["valueString"] = str(val)

        entries.append(observation)

    # --- 5. Procedures → Procedure (with SNOMED CT codings) ---
    grounded_procs = grounded_entities.get("procedures", [])
    raw_procs = extracted_data.get("procedures", [])

    for idx, proc_text in enumerate(raw_procs):
        g_info  = grounded_procs[idx] if idx < len(grounded_procs) else {}
        system  = g_info.get("system") or SYSTEM_SNOMED
        code    = g_info.get("code") or "UNMAPPED"
        display = g_info.get("display") or proc_text

        procedure: Dict[str, Any] = {
            "fullUrl": f"urn:uuid:{uuid.uuid4()}",
            "resource": {
                "resourceType": "Procedure",
                "status": "completed",
                "code": {
                    "coding": [{"system": system, "code": code, "display": display}],
                    "text": proc_text
                },
                "subject": {"reference": "Patient/1"}
            }
        }

        # Temporal: occurrence date
        occur_dt = _get_entity_datetime(proc_text, "procedures", temporal_timeline)
        if occur_dt and not occur_dt.startswith("on ") and not occur_dt.startswith("at "):
            procedure["resource"]["occurrenceDateTime"] = occur_dt

        entries.append(procedure)

    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries
    }

    return bundle
