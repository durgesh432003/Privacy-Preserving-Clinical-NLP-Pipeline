"""
Unit tests for 100% Local Deep Medical Ontology Grounding Engine and FHIR Enrichment.
"""

import pytest
from src.ontology.grounder import get_grounder, LocalOntologyGrounder
from src.agents.fhir_builder import build_fhir_bundle
from src.ontology.ontologies import SYSTEM_SNOMED, SYSTEM_RXNORM, SYSTEM_LOINC, SYSTEM_ICD10


def test_ground_diagnosis_snomed():
    grounder = get_grounder()
    concept = grounder.ground_entity("Essential Hypertension", category="diagnosis")
    
    assert concept.system == SYSTEM_SNOMED
    assert concept.code == "38341003"
    assert concept.display == "Hypertensive vascular disease"
    assert concept.icd10_code == "I10"
    assert concept.confidence > 0.8


def test_ground_medication_rxnorm():
    grounder = get_grounder()
    concept = grounder.ground_entity("Lisinopril 10mg", category="medication")
    
    assert concept.system == SYSTEM_RXNORM
    assert concept.code == "314076"
    assert "Lisinopril" in concept.display
    assert concept.confidence > 0.8


def test_ground_lab_loinc():
    grounder = get_grounder()
    concept = grounder.ground_entity("Hemoglobin A1c", category="lab")
    
    assert concept.system == SYSTEM_LOINC
    assert concept.code == "4548-4"
    assert concept.confidence > 0.8


def test_ground_procedure_snomed():
    grounder = get_grounder()
    concept = grounder.ground_entity("Echocardiogram", category="procedure")
    
    assert concept.system == SYSTEM_SNOMED
    assert concept.code == "40701008"
    assert concept.confidence > 0.8


def test_fhir_builder_with_grounded_ontologies():
    extracted = {
        "diagnoses": ["Hypertension", "Type 2 Diabetes"],
        "medications": [{"name": "Lisinopril 10mg", "dose": "10mg", "frequency": "once daily"}],
        "labs": [{"name": "Serum Creatinine", "value": "1.2", "unit": "mg/dL"}],
        "procedures": ["Echocardiogram"]
    }
    
    bundle = build_fhir_bundle(extracted)
    assert bundle["resourceType"] == "Bundle"
    assert len(bundle["entry"]) >= 4
    
    # Verify Condition entry has SNOMED and ICD-10 codings
    cond_entry = bundle["entry"][0]["resource"]
    assert cond_entry["resourceType"] == "Condition"
    codings = cond_entry["code"]["coding"]
    systems = [c["system"] for c in codings]
    assert SYSTEM_SNOMED in systems
    assert SYSTEM_ICD10 in systems
    
    # Verify MedicationStatement entry has RxNorm coding
    med_entry = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "MedicationStatement"][0]
    med_coding = med_entry["medication"]["concept"]["coding"][0]
    assert med_coding["system"] == SYSTEM_RXNORM
    assert med_coding["code"] == "314076"
    
    # Verify Observation entry has LOINC coding
    obs_entry = [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"][0]
    obs_coding = obs_entry["code"]["coding"][0]
    assert obs_coding["system"] == SYSTEM_LOINC
    assert obs_coding["code"] == "2160-0"
