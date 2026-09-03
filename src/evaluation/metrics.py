"""
Standardized Clinical NLP Evaluation Metrics Engine.

Implements publication-grade evaluation metrics for clinical entity extraction,
ontology grounding, and FHIR generation quality, following the conventions
of the n2c2 (i2b2) shared task competitions and ACL/EMNLP clinical NLP tracks.

Metrics implemented:
  1. Strict Entity F1     — Exact span + correct category (case-insensitive)
  2. Soft Entity F1       — Predicted entity is a substring of gold (or vice-versa)
  3. Concept Grounding Accuracy (Top-1, Top-3) — Ontology code match
  4. FHIR Schema Validity Score — Fraction of entries passing fhir.resources validation
  5. Assertion Classification Accuracy — PRESENT/ABSENT/POSSIBLE per entity
  6. Temporal Association Accuracy — Correct date/anchor for entity

All metric functions are stateless and composable.
Designed to accept gold-standard annotations compatible with n2c2 format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class EntityAnnotation:
    """A single annotated entity (gold or predicted)."""
    text: str           # raw entity text (e.g., "type 2 diabetes")
    category: str       # category key (e.g., "diagnoses", "medications")
    code: str = ""      # standard code if grounded (e.g., SNOMED "44054006")
    code_system: str = ""  # e.g., "http://snomed.info/sct"
    assertion: str = "PRESENT"   # PRESENT | ABSENT | POSSIBLE | HISTORICAL | FAMILY
    onset_date: Optional[str] = None  # ISO date string or named anchor


@dataclass
class EntityF1Result:
    """Token-level F1 evaluation result for one category."""
    category: str
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
        }


@dataclass
class EvaluationReport:
    """Aggregated evaluation report across all metric dimensions."""
    model_name: str = ""
    case_id: str = ""

    # Entity extraction
    strict_entity_f1: Dict[str, Any] = field(default_factory=dict)
    soft_entity_f1: Dict[str, Any] = field(default_factory=dict)
    macro_avg_strict_f1: float = 0.0
    macro_avg_soft_f1: float = 0.0

    # Grounding
    grounding_top1_accuracy: float = 0.0
    grounding_top3_accuracy: float = 0.0
    grounding_evaluated: int = 0

    # FHIR quality
    fhir_entry_validity_rate: float = 0.0
    fhir_total_entries: int = 0
    fhir_valid_entries: int = 0

    # Assertion accuracy
    assertion_accuracy: float = 0.0
    assertion_evaluated: int = 0

    # Temporal accuracy
    temporal_accuracy: float = 0.0
    temporal_evaluated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalize entity text for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_strict_match(pred: str, gold: str) -> bool:
    """Case-insensitive exact string match after normalization."""
    return _normalize(pred) == _normalize(gold)


def _is_soft_match(pred: str, gold: str) -> bool:
    """
    Soft span match: normalized pred is a substring of normalized gold,
    or normalized gold is a substring of normalized pred.
    This handles truncated vs. expanded mentions.
    """
    p, g = _normalize(pred), _normalize(gold)
    return p in g or g in p


# ---------------------------------------------------------------------------
# Entity F1 Evaluation
# ---------------------------------------------------------------------------

def compute_entity_f1(
    gold_entities: List[EntityAnnotation],
    pred_entities: List[EntityAnnotation],
    category: str,
    strict: bool = True,
) -> EntityF1Result:
    """
    Compute Precision, Recall, and F1 for a single entity category.

    Args:
        gold_entities: Gold-standard annotated entities (filtered to `category`)
        pred_entities: Predicted entities from the pipeline (filtered to `category`)
        category: Entity category string (e.g., "diagnoses")
        strict: If True, use exact match. If False, use substring soft match.
    """
    result = EntityF1Result(category=category)

    gold_filtered = [e for e in gold_entities if e.category == category]
    pred_filtered = [e for e in pred_entities if e.category == category]

    match_fn = _is_strict_match if strict else _is_soft_match

    gold_matched = [False] * len(gold_filtered)
    pred_matched = [False] * len(pred_filtered)

    for p_idx, pred in enumerate(pred_filtered):
        for g_idx, gold in enumerate(gold_filtered):
            if not gold_matched[g_idx] and match_fn(pred.text, gold.text):
                result.true_positive += 1
                gold_matched[g_idx] = True
                pred_matched[p_idx] = True
                break

    result.false_positive = sum(1 for m in pred_matched if not m)
    result.false_negative = sum(1 for m in gold_matched if not m)

    return result


def compute_all_entity_f1(
    gold_entities: List[EntityAnnotation],
    pred_entities: List[EntityAnnotation],
    categories: Optional[List[str]] = None,
    strict: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute entity F1 for all categories.
    Returns a dict: { "diagnoses": {...}, "medications": {...}, ... }
    """
    if categories is None:
        categories = list({e.category for e in gold_entities + pred_entities})

    results = {}
    for cat in categories:
        r = compute_entity_f1(gold_entities, pred_entities, cat, strict=strict)
        results[cat] = r.to_dict()

    return results


def compute_macro_avg_f1(per_category_results: Dict[str, Dict[str, Any]]) -> float:
    """Compute macro-averaged F1 across all categories."""
    f1_scores = [v["f1"] for v in per_category_results.values() if "f1" in v]
    return round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0


# ---------------------------------------------------------------------------
# Ontology Grounding Accuracy
# ---------------------------------------------------------------------------

def compute_grounding_accuracy(
    gold_entities: List[EntityAnnotation],
    pred_entities: List[EntityAnnotation],
    top_k: int = 1,
) -> Tuple[float, int]:
    """
    Compute ontology grounding accuracy.

    For each gold entity that has a code, check if the predicted entity's code
    matches. For top_k > 1, we accept any of the top-k predicted codes (simulated
    by checking if the predicted code matches any known synonym code).

    Returns: (accuracy, n_evaluated)
    """
    evaluated = 0
    correct = 0

    for gold in gold_entities:
        if not gold.code:
            continue  # Skip unannotated entities

        # Find matching predicted entity
        matched_pred = None
        for pred in pred_entities:
            if pred.category == gold.category and _is_soft_match(pred.text, gold.text):
                matched_pred = pred
                break

        if matched_pred is None:
            evaluated += 1  # Not found = wrong
            continue

        evaluated += 1

        # For top-1: exact code match
        pred_code = (matched_pred.code or "").strip()
        gold_code = (gold.code or "").strip()

        if pred_code == gold_code:
            correct += 1
        elif top_k > 1:
            # Soft code match: check if gold_code is a prefix/suffix of pred_code
            # (handles versioned codes like "J18" vs "J18.9")
            if gold_code.startswith(pred_code) or pred_code.startswith(gold_code):
                correct += 1

    accuracy = correct / evaluated if evaluated > 0 else 0.0
    return round(accuracy, 4), evaluated


# ---------------------------------------------------------------------------
# FHIR Schema Validity
# ---------------------------------------------------------------------------

def compute_fhir_validity(fhir_bundle: Optional[Dict[str, Any]]) -> Tuple[float, int, int]:
    """
    Validate each entry in a FHIR bundle against the fhir.resources library.

    Returns: (validity_rate, valid_count, total_count)
    """
    if not fhir_bundle:
        return 0.0, 0, 0

    entries = fhir_bundle.get("entry", [])
    total = len(entries)
    if total == 0:
        return 1.0, 0, 0

    valid_count = 0
    for entry in entries:
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "")
        try:
            from fhir.resources import construct_fhir_element
            construct_fhir_element(resource_type, resource)
            valid_count += 1
        except Exception:
            pass

    rate = valid_count / total
    return round(rate, 4), valid_count, total


# ---------------------------------------------------------------------------
# Assertion Classification Accuracy
# ---------------------------------------------------------------------------

def compute_assertion_accuracy(
    gold_entities: List[EntityAnnotation],
    pred_assertion_map: Dict[str, Dict[str, str]],
) -> Tuple[float, int]:
    """
    Compare gold assertion labels against predicted assertion_map.

    Returns: (accuracy, n_evaluated)
    """
    evaluated = 0
    correct = 0

    for gold in gold_entities:
        gold_assertion = (gold.assertion or "PRESENT").upper()
        category_map = pred_assertion_map.get(gold.category, {})

        # Find the matching key in assertion map (soft match)
        pred_assertion = "PRESENT"
        for entity_key, status in category_map.items():
            if _is_soft_match(entity_key, gold.text):
                pred_assertion = status.upper()
                break

        evaluated += 1
        if pred_assertion == gold_assertion:
            correct += 1

    accuracy = correct / evaluated if evaluated > 0 else 0.0
    return round(accuracy, 4), evaluated


# ---------------------------------------------------------------------------
# Temporal Association Accuracy
# ---------------------------------------------------------------------------

def compute_temporal_accuracy(
    gold_entities: List[EntityAnnotation],
    pred_timeline: Optional[Dict[str, Any]],
) -> Tuple[float, int]:
    """
    Check whether the predicted onset_date/occurrence_date for each entity
    matches the gold annotation.

    Returns: (accuracy, n_evaluated)
    """
    if not pred_timeline:
        return 0.0, 0

    entity_times = pred_timeline.get("entity_times", {})
    evaluated = 0
    correct = 0

    for gold in gold_entities:
        if not gold.onset_date:
            continue  # No temporal gold label — skip

        evaluated += 1
        category_times = entity_times.get(gold.category, {})

        pred_date = None
        for entity_key, dt in category_times.items():
            if _is_soft_match(entity_key, gold.text):
                pred_date = dt
                break

        if pred_date and pred_date == gold.onset_date:
            correct += 1

    accuracy = correct / evaluated if evaluated > 0 else 0.0
    return round(accuracy, 4), evaluated


# ---------------------------------------------------------------------------
# Full Report Generator
# ---------------------------------------------------------------------------

def generate_full_report(
    gold_entities: List[EntityAnnotation],
    pred_entities: List[EntityAnnotation],
    fhir_bundle: Optional[Dict[str, Any]] = None,
    pred_assertion_map: Optional[Dict[str, Dict[str, str]]] = None,
    pred_timeline: Optional[Dict[str, Any]] = None,
    model_name: str = "",
    case_id: str = "",
    categories: Optional[List[str]] = None,
) -> EvaluationReport:
    """
    Generate a comprehensive EvaluationReport covering all metric dimensions.
    """
    report = EvaluationReport(model_name=model_name, case_id=case_id)

    if categories is None:
        categories = list({e.category for e in gold_entities + pred_entities})

    # Entity F1 — Strict
    strict_results = compute_all_entity_f1(gold_entities, pred_entities, categories, strict=True)
    report.strict_entity_f1 = strict_results
    report.macro_avg_strict_f1 = compute_macro_avg_f1(strict_results)

    # Entity F1 — Soft
    soft_results = compute_all_entity_f1(gold_entities, pred_entities, categories, strict=False)
    report.soft_entity_f1 = soft_results
    report.macro_avg_soft_f1 = compute_macro_avg_f1(soft_results)

    # Grounding Accuracy
    top1_acc, n_ground = compute_grounding_accuracy(gold_entities, pred_entities, top_k=1)
    top3_acc, _ = compute_grounding_accuracy(gold_entities, pred_entities, top_k=3)
    report.grounding_top1_accuracy = top1_acc
    report.grounding_top3_accuracy = top3_acc
    report.grounding_evaluated = n_ground

    # FHIR Validity
    fhir_rate, fhir_valid_n, fhir_total = compute_fhir_validity(fhir_bundle)
    report.fhir_entry_validity_rate = fhir_rate
    report.fhir_valid_entries = fhir_valid_n
    report.fhir_total_entries = fhir_total

    # Assertion Accuracy
    if pred_assertion_map is not None:
        assert_acc, n_assert = compute_assertion_accuracy(gold_entities, pred_assertion_map)
        report.assertion_accuracy = assert_acc
        report.assertion_evaluated = n_assert

    # Temporal Accuracy
    if pred_timeline is not None:
        temp_acc, n_temp = compute_temporal_accuracy(gold_entities, pred_timeline)
        report.temporal_accuracy = temp_acc
        report.temporal_evaluated = n_temp

    return report


# ---------------------------------------------------------------------------
# Convenience: Build pred_entities from pipeline state dict
# ---------------------------------------------------------------------------

def pred_entities_from_state(state: Dict[str, Any]) -> List[EntityAnnotation]:
    """
    Convert pipeline state['extracted_data'] + state['assertion_map'] into
    a flat list of EntityAnnotation objects for evaluation.
    """
    extracted = state.get("extracted_data") or {}
    assertion_map = state.get("assertion_map") or {}
    grounded = state.get("grounded_entities") or {}

    entities: List[EntityAnnotation] = []

    for diag in extracted.get("diagnoses", []):
        g_list = grounded.get("diagnoses", [])
        code = g_list[0].get("code", "") if g_list else ""
        code_sys = g_list[0].get("system", "") if g_list else ""
        assertion = assertion_map.get("diagnoses", {}).get(diag, "PRESENT")
        entities.append(EntityAnnotation(
            text=diag,
            category="diagnoses",
            code=code,
            code_system=code_sys,
            assertion=assertion,
        ))

    for med in extracted.get("medications", []):
        name = med.get("name", "") if isinstance(med, dict) else str(med)
        assertion = assertion_map.get("medications", {}).get(name, "PRESENT")
        entities.append(EntityAnnotation(
            text=name,
            category="medications",
            assertion=assertion,
        ))

    for proc in extracted.get("procedures", []):
        assertion = assertion_map.get("procedures", {}).get(proc, "PRESENT")
        entities.append(EntityAnnotation(
            text=proc,
            category="procedures",
            assertion=assertion,
        ))

    return entities
