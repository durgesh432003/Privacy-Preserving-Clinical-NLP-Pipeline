"""
Clinical Assertion & Negation Detection Engine.

Implements a NegEx-inspired rule-based assertion classifier that labels each
extracted clinical entity as one of:

  - PRESENT      : Confirmed, active finding (maps to FHIR confirmed active Condition)
  - ABSENT       : Explicitly negated ("no chest pain", "denies fever")
  - POSSIBLE     : Uncertain / speculative ("r/o PE", "possible MI", "? angina")
  - FAMILY       : Family history attribution ("mother had breast cancer")
  - HISTORICAL   : Past history, not current ("history of appendectomy")
  - HYPOTHETICAL : Conditional framing ("if pain returns, consider …")

Design Goals:
  - Fully offline, zero ML inference, zero network calls.
  - O(n·m) rule scan per sentence, acceptable for short clinical notes.
  - Operates sentence-by-sentence to scope negation correctly.

References:
  Chapman WW et al., "A Simple Algorithm for Identifying Negated Findings and
  Diseases in Discharge Summaries." J Biomed Inform. 2001;34(5):301-10.
"""

import re
from enum import Enum
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Assertion Status Enum
# ---------------------------------------------------------------------------

class AssertionStatus(str, Enum):
    PRESENT    = "PRESENT"
    ABSENT     = "ABSENT"
    POSSIBLE   = "POSSIBLE"
    FAMILY     = "FAMILY"
    HISTORICAL = "HISTORICAL"
    HYPOTHETICAL = "HYPOTHETICAL"


# ---------------------------------------------------------------------------
# Trigger Lexicons
# ---------------------------------------------------------------------------

# Pre-negation triggers: negate entities that appear AFTER them in the sentence
PRE_NEGATION_TRIGGERS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bdenies\b",
    r"\bdenied\b",
    r"\bwithout\b",
    r"\bno evidence of\b",
    r"\bno signs of\b",
    r"\bno complaints of\b",
    r"\bfree of\b",
    r"\babsence of\b",
    r"\bnegative for\b",
    r"\bunremarkable\b",
    r"\bnon-contributory\b",
    r"\bno known\b",
    r"\bnot consistent with\b",
    r"\bdoes not have\b",
    r"\bhas not had\b",
    r"\bnever had\b",
]

# Post-negation triggers: negate entities that appear BEFORE them in the sentence
POST_NEGATION_TRIGGERS = [
    r"\bwas ruled out\b",
    r"\bhas been ruled out\b",
    r"\bnot present\b",
    r"\bnot seen\b",
    r"\bnot found\b",
    r"\bnot detected\b",
    r"\bwas excluded\b",
]

# Possibility / uncertainty triggers
# Note: "rule out" / "cannot rule out" are POSSIBLE, NOT negation.
# These are checked BEFORE negation triggers in classify_in_sentence.
POSSIBILITY_TRIGGERS = [
    r"\bpossible\b",
    r"\bprobable\b",
    r"\bprobably\b",
    r"\blikely\b",
    r"\bsuspected\b",
    r"\bsuspect\b",
    r"\br/o\b",
    r"\brule out\b",
    r"\bcannot rule out\b",
    r"\bcannot exclude\b",
    r"\bpossibly\b",
    r"\bappears to be\b",
    r"\bseems\b",
    r"\bmay have\b",
    r"\bmight have\b",
    r"\bcould be\b",
    r"\bquestionable\b",
    r"\bworrisome for\b",
    r"\bconcerning for\b",
    r"\bsuggestive of\b",
]

# Family history triggers
FAMILY_TRIGGERS = [
    r"\bfamily history\b",
    r"\bfh\b",
    r"\bmother had\b",
    r"\bfather had\b",
    r"\bbrother had\b",
    r"\bsister had\b",
    r"\bparent had\b",
    r"\bgrandmother had\b",
    r"\bgrandfather had\b",
    r"\bfamilial\b",
    r"\bheritable\b",
    r"\binherited\b",
]

# Historical triggers
HISTORICAL_TRIGGERS = [
    r"\bhistory of\b",
    r"\bh/o\b",
    r"\bpast medical history\b",
    r"\bpmh\b",
    r"\bpast history\b",
    r"\bpreviously\b",
    r"\bprevious\b",
    r"\bformer\b",
    r"\bstatus post\b",
    r"\bs/p\b",
    r"\bprior\b",
    r"\bprior history\b",
    r"\bold\b",
    r"\bin the past\b",
    r"\byears ago\b",
    r"\bmonths ago\b",
]

# Hypothetical / conditional triggers
# NOTE: "consider" removed from here because it more commonly signals POSSIBLE
HYPOTHETICAL_TRIGGERS = [
    r"\bif\b",
    r"\bshould\b",
    r"\bwould\b",
    r"\bin the event\b",
    r"\bcontingent on\b",
    r"\bshould symptoms\b",
    r"\bif he develops\b",
    r"\bif she develops\b",
    r"\bif patient develops\b",
    r"\bif symptoms\b",
    r"\bif pain\b",
    r"\bif (?:the )?patient\b",
]

# Termination words that stop the scope of a trigger
TERMINATION_TRIGGERS = [
    r"\bbut\b",
    r"\bhowever\b",
    r"\bexcept\b",
    r"\bother than\b",
    r"\bwith\b",
    r"\balthough\b",
    r"\bnevertheless\b",
    r"\bstill has\b",
    r"\bstill present\b",
]

# Max tokens of scope for a pre-negation trigger
PRE_NEGATION_SCOPE = 10   # words
POST_NEGATION_SCOPE = 5   # words (look behind)


# ---------------------------------------------------------------------------
# Compiled Patterns (module-level, computed once)
# ---------------------------------------------------------------------------

def _compile(patterns: List[str]) -> re.Pattern:
    return re.compile("|".join(patterns), flags=re.IGNORECASE)


_PRE_NEG_RE   = _compile(PRE_NEGATION_TRIGGERS)
_POST_NEG_RE  = _compile(POST_NEGATION_TRIGGERS)
_POSS_RE      = _compile(POSSIBILITY_TRIGGERS)
_FAMILY_RE    = _compile(FAMILY_TRIGGERS)
_HIST_RE      = _compile(HISTORICAL_TRIGGERS)
_HYPO_RE      = _compile(HYPOTHETICAL_TRIGGERS)
_TERM_RE      = _compile(TERMINATION_TRIGGERS)


# ---------------------------------------------------------------------------
# Core Classification Logic
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries for scoped detection."""
    # Split on period+space, semicolons, and newlines
    parts = re.split(r"(?<=[.;!\?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _entity_position(sentence: str, entity: str) -> Optional[int]:
    """Return character position of entity in sentence (case-insensitive), or None."""
    m = re.search(re.escape(entity), sentence, flags=re.IGNORECASE)
    return m.start() if m else None


def _word_distance(sentence: str, trigger_end: int, entity_start: int) -> int:
    """Count words between trigger end and entity start."""
    between = sentence[trigger_end:entity_start]
    return len(between.split())


def _has_termination_between(sentence: str, start: int, end: int) -> bool:
    """Check whether a termination trigger lies between start and end char positions."""
    between = sentence[start:end]
    return bool(_TERM_RE.search(between))


def classify_entity(text: str, entity: str) -> AssertionStatus:
    """
    Classify the assertion status of `entity` within the full `text` context.

    Processes sentence-by-sentence; the sentence that contains the entity
    determines the classification. Triggers in other sentences are ignored.
    """
    # Find the sentence containing the entity
    sentences = _split_sentences(text)
    containing_sentence = text  # fallback: whole text

    for sentence in sentences:
        if re.search(re.escape(entity), sentence, flags=re.IGNORECASE):
            containing_sentence = sentence
            break

    return classify_in_sentence(containing_sentence, entity)


def classify_in_sentence(sentence: str, entity: str) -> AssertionStatus:
    """
    Classify the assertion status of `entity` within a single sentence.

    Precedence (highest to lowest):
      FAMILY > HISTORICAL > POSSIBLE (incl. rule-out) > ABSENT (negation) > HYPOTHETICAL > PRESENT

    POSSIBLE is checked before ABSENT because phrases like
    "rule out X" and "cannot rule out X" express UNCERTAINTY — not negation.
    """
    entity_pos = _entity_position(sentence, entity)
    if entity_pos is None:
        return AssertionStatus.PRESENT

    entity_end = entity_pos + len(entity)

    # --- FAMILY HISTORY (checked first — highest semantic priority) ---
    for m in _FAMILY_RE.finditer(sentence):
        if m.start() < entity_pos:
            dist = _word_distance(sentence, m.end(), entity_pos)
            if dist <= 10 and not _has_termination_between(sentence, m.end(), entity_pos):
                return AssertionStatus.FAMILY
        else:
            dist = _word_distance(sentence, entity_end, m.start())
            if dist <= 5:
                return AssertionStatus.FAMILY

    # --- HISTORICAL ---
    for m in _HIST_RE.finditer(sentence):
        if m.start() < entity_pos:
            dist = _word_distance(sentence, m.end(), entity_pos)
            if dist <= 8 and not _has_termination_between(sentence, m.end(), entity_pos):
                return AssertionStatus.HISTORICAL

    # --- POSSIBLE (checked before ABSENT to correctly classify "rule out X" as POSSIBLE not ABSENT) ---
    for m in _POSS_RE.finditer(sentence):
        if m.start() < entity_pos:
            dist = _word_distance(sentence, m.end(), entity_pos)
            if dist <= 8 and not _has_termination_between(sentence, m.end(), entity_pos):
                return AssertionStatus.POSSIBLE

    # --- PRE-NEGATION (trigger before entity) ---
    for m in _PRE_NEG_RE.finditer(sentence):
        if m.end() > entity_pos:
            continue  # trigger is after entity — skip
        dist = _word_distance(sentence, m.end(), entity_pos)
        if dist > PRE_NEGATION_SCOPE:
            continue
        if _has_termination_between(sentence, m.end(), entity_pos):
            continue
        return AssertionStatus.ABSENT

    # --- POST-NEGATION (trigger after entity) ---
    for m in _POST_NEG_RE.finditer(sentence):
        if m.start() < entity_end:
            continue  # trigger is before entity — skip
        dist = _word_distance(sentence, entity_end, m.start())
        if dist > POST_NEGATION_SCOPE:
            continue
        if _has_termination_between(sentence, entity_end, m.start()):
            continue
        return AssertionStatus.ABSENT

    # --- HYPOTHETICAL ---
    for m in _HYPO_RE.finditer(sentence):
        if m.start() < entity_pos:
            dist = _word_distance(sentence, m.end(), entity_pos)
            if dist <= 10 and not _has_termination_between(sentence, m.end(), entity_pos):
                return AssertionStatus.HYPOTHETICAL

    return AssertionStatus.PRESENT


# ---------------------------------------------------------------------------
# Batch Classification
# ---------------------------------------------------------------------------

def classify_all_entities(text: str, extracted_data: dict) -> Dict[str, Dict[str, str]]:
    """
    Run assertion classification over all entities in extracted_data.

    Returns a nested dict:
    {
      "diagnoses": { "Chest pain": "ABSENT", "Pneumonia": "PRESENT", ... },
      "medications": { "Aspirin": "PRESENT", ... },
      "procedures": { ... },
    }
    """
    result: Dict[str, Dict[str, str]] = {
        "diagnoses": {},
        "medications": {},
        "procedures": {},
    }

    for diag in extracted_data.get("diagnoses", []):
        status = classify_entity(text, diag)
        result["diagnoses"][diag] = status.value

    for med in extracted_data.get("medications", []):
        name = med.get("name", "") if isinstance(med, dict) else str(med)
        if name:
            status = classify_entity(text, name)
            result["medications"][name] = status.value

    for proc in extracted_data.get("procedures", []):
        status = classify_entity(text, proc)
        result["procedures"][proc] = status.value

    return result


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------

def filter_present_only(entities: list, assertion_map: dict) -> list:
    """Filter a list of entity strings, keeping only those classified as PRESENT."""
    if not assertion_map:
        return entities
    return [e for e in entities if assertion_map.get(e, "PRESENT") == AssertionStatus.PRESENT.value]
