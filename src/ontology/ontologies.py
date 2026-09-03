"""
Local Clinical Terminology Database and System Definitions.
Provides curated, expandable dictionaries of SNOMED CT, RxNorm, LOINC, and ICD-10-CM terms.
Used for offline, zero-network medical entity grounding.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel

SYSTEM_SNOMED = "http://snomed.info/sct"
SYSTEM_RXNORM = "http://www.nlm.nih.gov/research/uil/rxnorm"
SYSTEM_LOINC = "http://loinc.org"
SYSTEM_ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"


class GroundedConcept(BaseModel):
    system: str
    code: str
    display: str
    category: str  # diagnosis, medication, lab, procedure
    synonyms: List[str] = []
    icd10_code: Optional[str] = None
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# LOCAL CLINICAL ONTOLOGY SEED DATA
# ---------------------------------------------------------------------------

DIAGNOSES_ONTOLOGY: List[GroundedConcept] = [
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="38341003",
        display="Hypertensive vascular disease",
        category="diagnosis",
        synonyms=["hypertension", "htn", "high blood pressure", "essential hypertension", "primary hypertension"],
        icd10_code="I10",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="44054006",
        display="Type 2 diabetes mellitus",
        category="diagnosis",
        synonyms=["type 2 diabetes", "t2dm", "diabetes mellitus type 2", "dm2", "niddm", "type ii diabetes"],
        icd10_code="E11.9",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="42343006",
        display="Congestive heart failure",
        category="diagnosis",
        synonyms=["chf", "congestive heart failure", "heart failure", "left ventricular failure"],
        icd10_code="I50.9",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="57054005",
        display="Acute myocardial infarction",
        category="diagnosis",
        synonyms=["myocardial infarction", "mi", "acute mi", "heart attack", "stemi", "nstemi", "acute coronary syndrome"],
        icd10_code="I21.9",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="233604007",
        display="Pneumonia",
        category="diagnosis",
        synonyms=["pneumonia", "pna", "community acquired pneumonia", "cap", "bacterial pneumonia", "lung infection"],
        icd10_code="J18.9",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="195967001",
        display="Asthma",
        category="diagnosis",
        synonyms=["asthma", "bronchial asthma", "reactive airway disease"],
        icd10_code="J45.909",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="13645005",
        display="Chronic obstructive lung disease",
        category="diagnosis",
        synonyms=["copd", "chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis"],
        icd10_code="J44.9",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="709044004",
        display="Chronic kidney disease",
        category="diagnosis",
        synonyms=["ckd", "chronic kidney disease", "chronic renal failure", "crf", "renal insufficiency"],
        icd10_code="N18.9",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="55822004",
        display="Hyperlipidemia",
        category="diagnosis",
        synonyms=["hyperlipidemia", "hld", "hypercholesterolemia", "high cholesterol", "dyslipidemia"],
        icd10_code="E78.5",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="49436004",
        display="Atrial fibrillation",
        category="diagnosis",
        synonyms=["atrial fibrillation", "afib", "a-fib", "af"],
        icd10_code="I48.91",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="68566005",
        display="Urinary tract infection",
        category="diagnosis",
        synonyms=["uti", "urinary tract infection", "cystitis", "pyelonephritis"],
        icd10_code="N39.0",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="91302008",
        display="Sepsis",
        category="diagnosis",
        synonyms=["sepsis", "septicemia", "septic shock", "systemic inflammatory response syndrome"],
        icd10_code="A41.9",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="230690007",
        display="Cerebrovascular accident",
        category="diagnosis",
        synonyms=["cva", "stroke", "ischemic stroke", "cerebrovascular accident", "brain attack"],
        icd10_code="I63.9",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="370143000",
        display="Major depressive disorder",
        category="diagnosis",
        synonyms=["depression", "mdd", "major depressive disorder", "clinical depression"],
        icd10_code="F32.9",
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="396275006",
        display="Osteoarthritis",
        category="diagnosis",
        synonyms=["osteoarthritis", "oa", "degenerative joint disease", "djd"],
        icd10_code="M19.90",
    ),
]


MEDICATIONS_ONTOLOGY: List[GroundedConcept] = [
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="314076",
        display="Lisinopril 10 MG Oral Tablet",
        category="medication",
        synonyms=["lisinopril", "prinivil", "zestril", "lisinopril 10mg", "lisinopril 20mg", "lisinopril 5mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="860975",
        display="Metformin hydrochloride 500 MG Oral Tablet",
        category="medication",
        synonyms=["metformin", "glucophage", "metformin hcl", "metformin 500mg", "metformin 1000mg", "metformin 850mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="617314",
        display="Atorvastatin 20 MG Oral Tablet",
        category="medication",
        synonyms=["atorvastatin", "lipitor", "atorvastatin calcium", "atorvastatin 20mg", "atorvastatin 40mg", "atorvastatin 80mg", "atorvastatin 10mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="197361",
        display="Amlodipine 5 MG Oral Tablet",
        category="medication",
        synonyms=["amlodipine", "norvasc", "amlodipine besylate", "amlodipine 5mg", "amlodipine 10mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="866514",
        display="Metoprolol Succinate 50 MG Extended Release Oral Tablet",
        category="medication",
        synonyms=["metoprolol", "toprol", "toprol xl", "metoprolol succinate", "metoprolol tartrate", "metoprolol 50mg", "metoprolol 25mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="243670",
        display="Aspirin 81 MG Oral Tablet",
        category="medication",
        synonyms=["aspirin", "asa", "baby aspirin", "aspirin 81mg", "aspirin 325mg", "ecotrin"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="310429",
        display="Furosemide 40 MG Oral Tablet",
        category="medication",
        synonyms=["furosemide", "lasix", "furosemide 40mg", "furosemide 20mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="312153",
        display="Omeprazole 20 MG Delayed Release Oral Capsule",
        category="medication",
        synonyms=["omeprazole", "prilosec", "omeprazole 20mg", "omeprazole 40mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="966244",
        display="Levothyroxine Sodium 50 MCG Oral Tablet",
        category="medication",
        synonyms=["levothyroxine", "synthroid", "levoxyl", "levothyroxine 50mcg", "levothyroxine 100mcg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="310465",
        display="Gabapentin 300 MG Oral Capsule",
        category="medication",
        synonyms=["gabapentin", "neurontin", "gabapentin 300mg", "gabapentin 100mg", "gabapentin 600mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="435",
        display="Albuterol Inhalation Solution",
        category="medication",
        synonyms=["albuterol", "ventolin", "proair", "provantil", "albuterol inhaler"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="723",
        display="Amoxicillin Oral Suspension / Capsule",
        category="medication",
        synonyms=["amoxicillin", "amoxil", "amoxicillin 500mg", "amoxicillin 875mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="5224",
        display="Losartan Potassium Oral Tablet",
        category="medication",
        synonyms=["losartan", "cozaar", "losartan 50mg", "losartan 100mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="5487",
        display="Hydrochlorothiazide Oral Tablet",
        category="medication",
        synonyms=["hydrochlorothiazide", "hctz", "microzide", "hydrochlorothiazide 25mg", "hydrochlorothiazide 12.5mg"],
    ),
    GroundedConcept(
        system=SYSTEM_RXNORM,
        code="11289",
        display="Warfarin Sodium Oral Tablet",
        category="medication",
        synonyms=["warfarin", "coumadin", "jantoven", "warfarin 5mg"],
    ),
]


LABS_ONTOLOGY: List[GroundedConcept] = [
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="4548-4",
        display="Hemoglobin A1c/Hemoglobin.total in Blood",
        category="lab",
        synonyms=["hemoglobin a1c", "hba1c", "a1c", "glycated hemoglobin"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="2160-0",
        display="Creatinine [Mass/volume] in Serum or Plasma",
        category="lab",
        synonyms=["creatinine", "serum creatinine", "scr", "cr"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="2093-3",
        display="Cholesterol [Mass/volume] in Serum or Plasma",
        category="lab",
        synonyms=["cholesterol", "total cholesterol", "serum cholesterol"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="2345-7",
        display="Glucose [Mass/volume] in Blood",
        category="lab",
        synonyms=["blood glucose", "glucose", "fasting glucose", "fbg", "bs"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="6690-2",
        display="Leukocytes [#/volume] in Blood by Automated count",
        category="lab",
        synonyms=["white blood cell count", "wbc", "leukocyte count", "white count"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="718-7",
        display="Hemoglobin [Mass/volume] in Blood",
        category="lab",
        synonyms=["hemoglobin", "hgb", "hb"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="777-3",
        display="Platelets [#/volume] in Blood by Automated count",
        category="lab",
        synonyms=["platelet count", "platelets", "plt"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="2823-3",
        display="Potassium [Moles/volume] in Serum or Plasma",
        category="lab",
        synonyms=["potassium", "serum potassium", "k+"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="2951-2",
        display="Sodium [Moles/volume] in Serum or Plasma",
        category="lab",
        synonyms=["sodium", "serum sodium", "na+"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="85354-9",
        display="Blood pressure panel with all children optional",
        category="lab",
        synonyms=["blood pressure", "bp", "systolic blood pressure", "diastolic blood pressure"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="10839-9",
        display="Troponin I.cardiac [Mass/volume] in Serum or Plasma",
        category="lab",
        synonyms=["troponin", "troponin i", "cardiac troponin", "c-tni"],
    ),
    GroundedConcept(
        system=SYSTEM_LOINC,
        code="3016-3",
        display="Thyroid stimulating hormone [Units/volume] in Serum or Plasma",
        category="lab",
        synonyms=["tsh", "thyroid stimulating hormone", "thyrotropin"],
    ),
]


PROCEDURES_ONTOLOGY: List[GroundedConcept] = [
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="40701008",
        display="Echocardiography",
        category="procedure",
        synonyms=["echocardiogram", "echo", "transthoracic echocardiogram", "tte", "cardiac ultrasound"],
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="29303009",
        display="Electrocardiogram",
        category="procedure",
        synonyms=["electrocardiogram", "ecg", "ekg", "12-lead ecg", "12 lead ekg"],
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="168731009",
        display="Standard chest X-ray",
        category="procedure",
        synonyms=["chest x-ray", "cxr", "chest radiograph", "chest x ray"],
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="168537006",
        display="Computed tomography of chest",
        category="procedure",
        synonyms=["ct chest", "ct scan of chest", "chest ct", "computed tomography chest"],
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="33367005",
        display="Coronary angiography",
        category="procedure",
        synonyms=["coronary angiogram", "cardiac cath", "cardiac catheterization", "coronary angiography"],
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="302497006",
        display="Hemodialysis",
        category="procedure",
        synonyms=["hemodialysis", "dialysis", "hd"],
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="73761001",
        display="Colonoscopy",
        category="procedure",
        synonyms=["colonoscopy", "screening colonoscopy", "lower endoscopy"],
    ),
    GroundedConcept(
        system=SYSTEM_SNOMED,
        code="91251008",
        display="Physical therapy procedure",
        category="procedure",
        synonyms=["physical therapy", "pt", "physiotherapy", "rehabilitation"],
    ),
]


ALL_ONTOLOGIES: Dict[str, List[GroundedConcept]] = {
    "diagnosis": DIAGNOSES_ONTOLOGY,
    "medication": MEDICATIONS_ONTOLOGY,
    "lab": LABS_ONTOLOGY,
    "procedure": PROCEDURES_ONTOLOGY,
}
