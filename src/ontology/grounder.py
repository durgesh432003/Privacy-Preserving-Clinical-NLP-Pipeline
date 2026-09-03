import re
from typing import Dict, List, Any, Optional, Tuple
from src.ontology.ontologies import (
    ALL_ONTOLOGIES,
    GroundedConcept,
    SYSTEM_SNOMED,
    SYSTEM_RXNORM,
    SYSTEM_LOINC,
    SYSTEM_ICD10,
)


def _clean_text(text: str) -> str:
    """Normalize text for semantic comparison."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> set:
    return set(_clean_text(text).split())


def _char_ngrams(text: str, n: int = 3) -> set:
    clean = _clean_text(text)
    if len(clean) < n:
        return {clean}
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def _jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


class LocalOntologyGrounder:
    """
    Offline clinical ontology grounding engine.
    Matches extracted medical entities against local SNOMED CT, RxNorm, LOINC, and ICD-10 datasets.
    """

    def __init__(self, ontologies: Optional[Dict[str, List[GroundedConcept]]] = None):
        self.ontologies = ontologies or ALL_ONTOLOGIES

    def ground_entity(
        self, query: str, category: str = "diagnosis"
    ) -> GroundedConcept:
        """
        Ground a single extracted text term to a formal ontology concept.
        Returns a GroundedConcept with a confidence score (0.0 to 1.0).
        """
        clean_query = _clean_text(query)
        if not clean_query:
            return GroundedConcept(
                system="http://terminology.hl7.org/CodeSystem/v3-NullFlavor",
                code="UNK",
                display=query or "Unknown",
                category=category,
                confidence=0.0,
            )

        # Get relevant ontology list for this category
        target_list = self.ontologies.get(category, [])
        if not target_list:
            # Fallback across all categories
            target_list = [
                item for sublist in self.ontologies.values() for item in sublist
            ]

        best_concept: Optional[GroundedConcept] = None
        best_score: float = 0.0

        query_tokens = _tokenize(query)
        query_ngrams = _char_ngrams(query)

        for concept in target_list:
            # 1. Exact match on display or synonyms -> score = 1.0
            all_terms = [concept.display] + concept.synonyms
            for term in all_terms:
                clean_term = _clean_text(term)
                if clean_query == clean_term:
                    score = 1.0
                elif clean_query in clean_term or clean_term in clean_query:
                    score = 0.85
                else:
                    term_tokens = _tokenize(term)
                    token_sim = _jaccard_similarity(query_tokens, term_tokens)
                    term_ngrams = _char_ngrams(term)
                    ngram_sim = _jaccard_similarity(query_ngrams, term_ngrams)
                    score = (token_sim * 0.6) + (ngram_sim * 0.4)

                if score > best_score:
                    best_score = score
                    best_concept = concept

        if best_concept and best_score >= 0.25:
            # Return clone of concept with calibrated confidence score
            result = best_concept.model_copy()
            result.confidence = round(best_score, 3)
            return result

        # Fallback concept if below confidence threshold
        default_system = (
            SYSTEM_SNOMED
            if category in ["diagnosis", "procedure"]
            else (SYSTEM_RXNORM if category == "medication" else SYSTEM_LOINC)
        )

        return GroundedConcept(
            system=default_system,
            code="UNMAPPED",
            display=query,
            category=category,
            confidence=round(best_score, 3),
        )

    def ground_extracted_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ground all entities inside ExtractedData dictionary.
        Returns a dictionary of grounded concept objects categorized by type.
        """
        grounded_results = {
            "diagnoses": [],
            "medications": [],
            "labs": [],
            "procedures": [],
        }

        # 1. Diagnoses
        for diag in extracted_data.get("diagnoses", []):
            grounded = self.ground_entity(diag, category="diagnosis")
            grounded_results["diagnoses"].append(grounded.model_dump())

        # 2. Medications
        for med in extracted_data.get("medications", []):
            name = med.get("name", "") if isinstance(med, dict) else str(med)
            grounded = self.ground_entity(name, category="medication")
            dump = grounded.model_dump()
            if isinstance(med, dict):
                dump["dose"] = med.get("dose")
                dump["frequency"] = med.get("frequency")
            grounded_results["medications"].append(dump)

        # 3. Procedures
        for proc in extracted_data.get("procedures", []):
            grounded = self.ground_entity(proc, category="procedure")
            grounded_results["procedures"].append(grounded.model_dump())

        # 4. Labs / Observations
        for lab in extracted_data.get("labs", []) or []:
            name = lab.get("name", "") if isinstance(lab, dict) else str(lab)
            grounded = self.ground_entity(name, category="lab")
            dump = grounded.model_dump()
            if isinstance(lab, dict):
                dump["value"] = lab.get("value")
                dump["unit"] = lab.get("unit")
            grounded_results["labs"].append(dump)

        return grounded_results


# Global grounder singleton
_grounder_instance: Optional[LocalOntologyGrounder] = None


def get_grounder() -> LocalOntologyGrounder:
    global _grounder_instance
    if _grounder_instance is None:
        _grounder_instance = LocalOntologyGrounder()
    return _grounder_instance
