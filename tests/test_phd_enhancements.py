"""
Comprehensive tests for PhD-level clinical NLP enhancements.

Tests cover:
1. Assertion & Negation Detection (NegEx engine)
2. Temporal Expression Extraction
3. Evaluation Metrics (Entity F1, Grounding Accuracy, Assertion Accuracy)

All tests are unit tests — zero network calls, zero LLM inference.
Run with: pytest tests/test_phd_enhancements.py -v
"""

import pytest
from src.nlp.assertion_detector import (
    AssertionStatus,
    classify_entity,
    classify_in_sentence,
    classify_all_entities,
)
from src.nlp.temporal_extractor import (
    TemporalExtractor,
    build_temporal_timeline,
)
from src.evaluation.metrics import (
    EntityAnnotation,
    EntityF1Result,
    compute_entity_f1,
    compute_all_entity_f1,
    compute_macro_avg_f1,
    compute_grounding_accuracy,
    compute_assertion_accuracy,
    generate_full_report,
    pred_entities_from_state,
    _is_strict_match,
    _is_soft_match,
)


# ============================================================
# 1. ASSERTION & NEGATION DETECTION TESTS
# ============================================================

class TestAssertionDetector:

    # --- Basic PRESENT (no trigger) ---
    def test_present_no_trigger(self):
        text = "Patient was admitted with pneumonia and started on antibiotics."
        assert classify_entity(text, "pneumonia") == AssertionStatus.PRESENT

    def test_present_medication(self):
        text = "Patient is prescribed Metformin 500mg twice daily."
        assert classify_entity(text, "Metformin") == AssertionStatus.PRESENT

    # --- ABSENT / Negation ---
    def test_negated_no(self):
        text = "Patient denies chest pain."
        assert classify_entity(text, "chest pain") == AssertionStatus.ABSENT

    def test_negated_denies(self):
        text = "He denies any shortness of breath."
        assert classify_entity(text, "shortness of breath") == AssertionStatus.ABSENT

    def test_negated_no_evidence(self):
        text = "No evidence of pulmonary embolism on CT scan."
        assert classify_entity(text, "pulmonary embolism") == AssertionStatus.ABSENT

    def test_negated_ruled_out(self):
        text = "Pulmonary embolism was ruled out."
        assert classify_entity(text, "Pulmonary embolism") == AssertionStatus.ABSENT

    def test_negated_absence_of(self):
        text = "Absence of pneumothorax was confirmed on chest X-ray."
        assert classify_entity(text, "pneumothorax") == AssertionStatus.ABSENT

    def test_negated_negative_for(self):
        text = "Patient was negative for COVID-19 infection."
        assert classify_entity(text, "COVID-19 infection") == AssertionStatus.ABSENT

    def test_negated_without(self):
        text = "Patient was discharged without fever."
        assert classify_entity(text, "fever") == AssertionStatus.ABSENT

    # --- POSSIBLE / Uncertain ---
    def test_possible_rule_out(self):
        text = "Rule out acute MI given elevated troponin."
        # 'rule out' is a POSSIBILITY trigger — not negation
        result = classify_entity(text, "acute MI")
        assert result == AssertionStatus.POSSIBLE

    def test_possible_possible(self):
        text = "Possible congestive heart failure."
        assert classify_entity(text, "congestive heart failure") == AssertionStatus.POSSIBLE

    def test_possible_concerning_for(self):
        text = "Imaging concerning for lung malignancy."
        assert classify_entity(text, "lung malignancy") == AssertionStatus.POSSIBLE

    def test_possible_cannot_rule_out(self):
        text = "Cannot rule out sepsis at this time."
        # 'cannot rule out' is a POSSIBILITY trigger
        result = classify_entity(text, "sepsis")
        assert result == AssertionStatus.POSSIBLE

    # --- FAMILY ---
    def test_family_history(self):
        text = "Family history significant for breast cancer in her mother."
        assert classify_entity(text, "breast cancer") == AssertionStatus.FAMILY

    def test_family_father(self):
        text = "Father had coronary artery disease."
        assert classify_entity(text, "coronary artery disease") == AssertionStatus.FAMILY

    # --- HISTORICAL ---
    def test_historical_history_of(self):
        text = "Patient has a history of atrial fibrillation."
        assert classify_entity(text, "atrial fibrillation") == AssertionStatus.HISTORICAL

    def test_historical_hx(self):
        text = "H/O appendectomy 5 years ago."
        assert classify_entity(text, "appendectomy") == AssertionStatus.HISTORICAL

    def test_historical_status_post(self):
        text = "Status post CABG in 2019."
        assert classify_entity(text, "CABG") == AssertionStatus.HISTORICAL

    # --- Scope termination (negation should NOT cross "but") ---
    def test_negation_terminated_by_but(self):
        text = "No fever, but the patient does have significant chest pain."
        # "chest pain" follows "but" — negation scope is terminated
        assert classify_entity(text, "chest pain") == AssertionStatus.PRESENT

    # --- HYPOTHETICAL ---
    def test_hypothetical_if(self):
        # Uses a specific if-patient phrasing which is a HYPOTHETICAL trigger
        text = "If the patient develops worsening symptoms, consider pulmonary embolism workup."
        # 'if the patient' fires HYPOTHETICAL before 'consider' fires POSSIBLE
        result = classify_entity(text, "pulmonary embolism")
        assert result == AssertionStatus.HYPOTHETICAL

    # --- Batch classification ---
    def test_classify_all_entities(self):
        text = (
            "Patient has pneumonia. No chest pain. "
            "Possible CHF. Family history of MI. H/O atrial fibrillation."
        )
        extracted = {
            "diagnoses": ["pneumonia", "chest pain", "CHF", "MI", "atrial fibrillation"],
            "medications": [],
            "procedures": [],
        }
        result = classify_all_entities(text, extracted)
        assert result["diagnoses"]["pneumonia"] == "PRESENT"
        assert result["diagnoses"]["chest pain"] == "ABSENT"
        assert result["diagnoses"]["CHF"] == "POSSIBLE"
        assert result["diagnoses"]["MI"] == "FAMILY"
        assert result["diagnoses"]["atrial fibrillation"] == "HISTORICAL"


# ============================================================
# 2. TEMPORAL EXPRESSION EXTRACTION TESTS
# ============================================================

class TestTemporalExtractor:
    def setup_method(self):
        from datetime import date
        self.extractor = TemporalExtractor(reference_date=date(2024, 3, 20))

    def test_iso_date(self):
        expressions = self.extractor.extract("Admitted on 2024-03-15.")
        assert len(expressions) == 1
        assert expressions[0].normalized_date == "2024-03-15"
        assert expressions[0].temporal_type == "absolute"

    def test_us_date(self):
        expressions = self.extractor.extract("Admitted on 03/15/2024.")
        assert any(e.normalized_date == "2024-03-15" for e in expressions)

    def test_written_date(self):
        expressions = self.extractor.extract("Admitted on January 15, 2024.")
        assert any(e.normalized_date == "2024-01-15" for e in expressions)

    def test_relative_ago(self):
        expressions = self.extractor.extract("Patient developed symptoms 3 days ago.")
        relative = [e for e in expressions if e.temporal_type == "relative"]
        assert len(relative) >= 1
        # 3 days before 2024-03-20 = 2024-03-17
        assert any("2024-03-17" in (e.normalized_date or "") for e in relative)

    def test_relative_prior(self):
        expressions = self.extractor.extract("Chest pain started 2 weeks prior to admission.")
        relative = [e for e in expressions if e.temporal_type == "relative"]
        assert len(relative) >= 1

    def test_named_anchor_on_admission(self):
        expressions = self.extractor.extract("Chest X-ray was taken on admission.")
        anchors = [e for e in expressions if e.temporal_type == "anchor"]
        assert len(anchors) >= 1
        assert any("on admission" in (e.normalized_date or "").lower() for e in anchors)

    def test_duration(self):
        expressions = self.extractor.extract("Patient has had hypertension for the past 5 years.")
        durations = [e for e in expressions if e.temporal_type == "duration"]
        assert len(durations) >= 1

    def test_entity_association(self):
        text = "Patient was admitted on 2024-03-10 with pneumonia."
        expressions = self.extractor.extract(text)
        assoc = self.extractor.associate_entity_to_time(text, "pneumonia", expressions)
        assert assoc is not None
        assert assoc.normalized_date == "2024-03-10"

    def test_build_temporal_timeline(self):
        text = "Patient admitted on 2024-01-15 with chest pain. Aspirin started on admission."
        extracted = {
            "diagnoses": ["chest pain"],
            "medications": [{"name": "Aspirin"}],
            "procedures": [],
        }
        timeline = build_temporal_timeline(text, extracted)
        assert "expressions" in timeline
        assert "entity_times" in timeline
        assert timeline["entity_times"]["diagnoses"]["chest pain"] == "2024-01-15"

    def test_no_date_in_text(self):
        text = "Patient has diabetes and hypertension."
        extracted = {"diagnoses": ["diabetes"], "medications": [], "procedures": []}
        timeline = build_temporal_timeline(text, extracted)
        # No dates → entity_time should be None
        assert timeline["entity_times"]["diagnoses"]["diabetes"] is None


# ============================================================
# 3. EVALUATION METRICS TESTS
# ============================================================

class TestEvaluationMetrics:

    def _make_gold(self):
        return [
            EntityAnnotation(text="pneumonia", category="diagnoses", code="233604007",
                             code_system="http://snomed.info/sct", assertion="PRESENT"),
            EntityAnnotation(text="hypertension", category="diagnoses", code="38341003",
                             code_system="http://snomed.info/sct", assertion="HISTORICAL"),
            EntityAnnotation(text="chest pain", category="diagnoses", assertion="ABSENT"),
            EntityAnnotation(text="Metformin", category="medications", code="6809",
                             code_system="http://www.nlm.nih.gov/research/uil/rxnorm", assertion="PRESENT"),
        ]

    def _make_pred_perfect(self):
        return [
            EntityAnnotation(text="pneumonia", category="diagnoses", code="233604007",
                             code_system="http://snomed.info/sct"),
            EntityAnnotation(text="hypertension", category="diagnoses", code="38341003",
                             code_system="http://snomed.info/sct"),
            EntityAnnotation(text="chest pain", category="diagnoses"),
            EntityAnnotation(text="Metformin", category="medications", code="6809",
                             code_system="http://www.nlm.nih.gov/research/uil/rxnorm"),
        ]

    def _make_pred_partial(self):
        return [
            EntityAnnotation(text="pneumonia", category="diagnoses", code="233604007"),
            # Missing: hypertension, chest pain
            EntityAnnotation(text="Metformin", category="medications", code="6809"),
            # Extra (FP):
            EntityAnnotation(text="appendicitis", category="diagnoses"),
        ]

    # --- Strict Entity F1 ---
    def test_strict_f1_perfect(self):
        gold = self._make_gold()
        pred = self._make_pred_perfect()
        result = compute_entity_f1(gold, pred, "diagnoses", strict=True)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_strict_f1_partial(self):
        gold = self._make_gold()
        pred = self._make_pred_partial()
        result = compute_entity_f1(gold, pred, "diagnoses", strict=True)
        assert result.true_positive == 1   # only "pneumonia" matched
        assert result.false_positive == 1  # "appendicitis" is extra
        assert result.false_negative == 2  # missing "hypertension", "chest pain"
        assert 0.0 < result.f1 < 1.0

    def test_strict_f1_empty_pred(self):
        gold = self._make_gold()
        result = compute_entity_f1(gold, [], "diagnoses", strict=True)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_strict_f1_empty_gold(self):
        pred = self._make_pred_perfect()
        result = compute_entity_f1([], pred, "diagnoses", strict=True)
        assert result.precision == 0.0
        assert result.recall == 0.0  # No gold = 0 denom
        assert result.f1 == 0.0

    # --- Soft Entity F1 ---
    def test_soft_f1_partial_string(self):
        gold = [EntityAnnotation(text="type 2 diabetes mellitus", category="diagnoses")]
        pred = [EntityAnnotation(text="type 2 diabetes", category="diagnoses")]
        result = compute_entity_f1(gold, pred, "diagnoses", strict=False)
        assert result.true_positive == 1  # soft match: "type 2 diabetes" in "type 2 diabetes mellitus"

    def test_soft_f1_strict_would_fail(self):
        gold = [EntityAnnotation(text="type 2 diabetes mellitus", category="diagnoses")]
        pred = [EntityAnnotation(text="type 2 diabetes", category="diagnoses")]
        result = compute_entity_f1(gold, pred, "diagnoses", strict=True)
        assert result.true_positive == 0  # exact match fails

    # --- String matching helpers ---
    def test_is_strict_match_case_insensitive(self):
        assert _is_strict_match("Pneumonia", "pneumonia")
        assert _is_strict_match("TYPE 2 DIABETES", "type 2 diabetes")

    def test_is_soft_match_substring(self):
        assert _is_soft_match("type 2 diabetes", "type 2 diabetes mellitus")
        assert _is_soft_match("type 2 diabetes mellitus", "type 2 diabetes")

    def test_is_soft_match_no_match(self):
        assert not _is_soft_match("pneumonia", "hypertension")

    # --- Macro average F1 ---
    def test_macro_avg_f1(self):
        results = {
            "diagnoses": {"f1": 0.80},
            "medications": {"f1": 0.60},
            "procedures": {"f1": 0.70},
        }
        avg = compute_macro_avg_f1(results)
        assert abs(avg - 0.7) < 0.001

    # --- Grounding accuracy ---
    def test_grounding_top1_perfect(self):
        gold = [EntityAnnotation(text="pneumonia", category="diagnoses", code="233604007")]
        pred = [EntityAnnotation(text="pneumonia", category="diagnoses", code="233604007")]
        acc, n = compute_grounding_accuracy(gold, pred, top_k=1)
        assert acc == 1.0
        assert n == 1

    def test_grounding_top1_wrong_code(self):
        gold = [EntityAnnotation(text="pneumonia", category="diagnoses", code="233604007")]
        pred = [EntityAnnotation(text="pneumonia", category="diagnoses", code="999999999")]
        acc, n = compute_grounding_accuracy(gold, pred, top_k=1)
        assert acc == 0.0
        assert n == 1

    def test_grounding_unannotated_gold_skipped(self):
        gold = [EntityAnnotation(text="pneumonia", category="diagnoses", code="")]
        pred = [EntityAnnotation(text="pneumonia", category="diagnoses", code="233604007")]
        acc, n = compute_grounding_accuracy(gold, pred, top_k=1)
        assert n == 0  # no gold codes to evaluate

    # --- Assertion accuracy ---
    def test_assertion_accuracy_perfect(self):
        gold = [
            EntityAnnotation(text="pneumonia", category="diagnoses", assertion="PRESENT"),
            EntityAnnotation(text="chest pain", category="diagnoses", assertion="ABSENT"),
        ]
        pred_map = {"diagnoses": {"pneumonia": "PRESENT", "chest pain": "ABSENT"}}
        acc, n = compute_assertion_accuracy(gold, pred_map)
        assert acc == 1.0
        assert n == 2

    def test_assertion_accuracy_partial(self):
        gold = [
            EntityAnnotation(text="pneumonia", category="diagnoses", assertion="PRESENT"),
            EntityAnnotation(text="chest pain", category="diagnoses", assertion="ABSENT"),
        ]
        pred_map = {"diagnoses": {"pneumonia": "PRESENT", "chest pain": "PRESENT"}}  # wrong assertion for chest pain
        acc, n = compute_assertion_accuracy(gold, pred_map)
        assert acc == 0.5
        assert n == 2

    # --- Full report ---
    def test_full_report_generates(self):
        gold = self._make_gold()
        pred = self._make_pred_perfect()
        report = generate_full_report(
            gold_entities=gold,
            pred_entities=pred,
            model_name="test-model",
            case_id="TEST-001",
        )
        assert report.macro_avg_strict_f1 > 0.0
        assert report.macro_avg_soft_f1 > 0.0
        assert "diagnoses" in report.strict_entity_f1

    def test_pred_entities_from_state(self):
        state = {
            "extracted_data": {
                "diagnoses": ["pneumonia", "chest pain"],
                "medications": [{"name": "Aspirin", "dose": "100mg", "frequency": "daily"}],
                "procedures": ["chest X-ray"],
            },
            "assertion_map": {
                "diagnoses": {"pneumonia": "PRESENT", "chest pain": "ABSENT"},
                "medications": {"Aspirin": "PRESENT"},
                "procedures": {"chest X-ray": "PRESENT"},
            },
            "grounded_entities": {},
        }
        entities = pred_entities_from_state(state)
        assert len(entities) == 4
        diag_names = [e.text for e in entities if e.category == "diagnoses"]
        assert "pneumonia" in diag_names
        assertion_map_check = {e.text: e.assertion for e in entities if e.category == "diagnoses"}
        assert assertion_map_check["chest pain"] == "ABSENT"
