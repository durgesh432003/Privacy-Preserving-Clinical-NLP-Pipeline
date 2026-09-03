"""
Temporal Expression Extractor.

Extracts and normalizes temporal expressions from clinical text, building a
timeline of medical events. This enriches FHIR resources with:
  - onsetDateTime  on Condition resources
  - occurrenceDateTime on Procedure resources
  - effectivePeriod / effectiveDateTime on Observation resources

Supports:
  - Absolute dates: "January 12, 2024", "01/12/2024", "2024-01-12"
  - Relative expressions: "3 days prior to admission", "post-op day 2", "2 weeks ago"
  - Named anchors: "on admission", "at discharge", "upon presentation"
  - Duration spans: "for 3 months", "over the past year"

Design: Fully offline, zero network calls, zero ML inference.
"""

import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple


# ---------------------------------------------------------------------------
# Pattern Definitions
# ---------------------------------------------------------------------------

# ISO / Numeric absolute dates
_ISO_DATE_RE = re.compile(
    r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b"
)
_US_DATE_RE = re.compile(
    r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"
)
_WRITTEN_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    flags=re.IGNORECASE,
)

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Relative expressions
_RELATIVE_AGO_RE = re.compile(
    r"\b(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago\b",
    flags=re.IGNORECASE,
)
_RELATIVE_PRIOR_RE = re.compile(
    r"\b(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+(?:prior|before|earlier)\b",
    flags=re.IGNORECASE,
)
_POST_OP_RE = re.compile(
    r"\bpost[-\s]?op(?:erative)?\s+day\s+(\d+)\b",
    flags=re.IGNORECASE,
)

# Named anchor expressions
_NAMED_ANCHOR_RE = re.compile(
    r"\b(on admission|at admission|upon admission|on presentation|at presentation|upon presentation"
    r"|at discharge|upon discharge|on discharge"
    r"|during (?:this )?(?:hospitalization|admission|stay)"
    r"|in (?:the )?(?:emergency|ED|ER)"
    r")\b",
    flags=re.IGNORECASE,
)

# Duration patterns
_DURATION_RE = re.compile(
    r"\b(?:for|over)\s+(?:the\s+)?(?:past\s+)?(\d+)\s+(day|days|week|weeks|month|months|year|years)\b",
    flags=re.IGNORECASE,
)

# Entity–time association patterns (entity mention near a temporal expression)
_TIME_OF_ONSET_CONTEXT_RE = re.compile(
    r"\b(?:since|starting|began|beginning|onset|started|developed|experienced|noticed)\b",
    flags=re.IGNORECASE,
)

_PROCEDURE_TIME_CONTEXT_RE = re.compile(
    r"\b(?:underwent|performed|done|completed|had|received)\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Normalization Helpers
# ---------------------------------------------------------------------------

def _to_unit_days(n: int, unit: str) -> int:
    """Convert (n, unit) to approximate days."""
    unit = unit.lower().rstrip("s")
    if unit == "day":
        return n
    if unit == "week":
        return n * 7
    if unit == "month":
        return n * 30
    if unit == "year":
        return n * 365
    return n


def _normalize_to_iso(dt: date) -> str:
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Temporal Expression Data Classes
# ---------------------------------------------------------------------------

class TemporalExpression:
    """Represents a single extracted temporal expression."""
    __slots__ = ("raw_text", "normalized_date", "temporal_type", "char_start", "char_end", "context")

    def __init__(
        self,
        raw_text: str,
        normalized_date: Optional[str],  # ISO 8601 or named anchor string
        temporal_type: str,              # absolute | relative | anchor | duration
        char_start: int,
        char_end: int,
        context: str = "",
    ):
        self.raw_text = raw_text
        self.normalized_date = normalized_date
        self.temporal_type = temporal_type
        self.char_start = char_start
        self.char_end = char_end
        self.context = context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_date": self.normalized_date,
            "temporal_type": self.temporal_type,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "context": self.context,
        }


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class TemporalExtractor:
    """
    Extract and normalize temporal expressions from clinical text.
    Reference date defaults to today (representing document processing date).
    """

    def __init__(self, reference_date: Optional[date] = None):
        self.reference_date = reference_date or date.today()

    def extract(self, text: str) -> List[TemporalExpression]:
        """Extract all temporal expressions from `text`, sorted by position."""
        expressions: List[TemporalExpression] = []

        # 1. ISO / numeric absolute dates
        for m in _ISO_DATE_RE.finditer(text):
            try:
                yr, mo, dy = int(m.group(1)), int(m.group(2)), int(m.group(3))
                dt = date(yr, mo, dy)
                expressions.append(TemporalExpression(
                    raw_text=m.group(),
                    normalized_date=_normalize_to_iso(dt),
                    temporal_type="absolute",
                    char_start=m.start(),
                    char_end=m.end(),
                ))
            except ValueError:
                pass

        # 2. US-style numeric dates (MM/DD/YYYY)
        for m in _US_DATE_RE.finditer(text):
            try:
                mo, dy, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
                dt = date(yr, mo, dy)
                expressions.append(TemporalExpression(
                    raw_text=m.group(),
                    normalized_date=_normalize_to_iso(dt),
                    temporal_type="absolute",
                    char_start=m.start(),
                    char_end=m.end(),
                ))
            except ValueError:
                pass

        # 3. Written-out dates
        for m in _WRITTEN_DATE_RE.finditer(text):
            try:
                mo = _MONTH_MAP[m.group(1).lower()]
                dy = int(re.sub(r"[^\d]", "", m.group(2)))
                yr = int(m.group(3))
                dt = date(yr, mo, dy)
                expressions.append(TemporalExpression(
                    raw_text=m.group(),
                    normalized_date=_normalize_to_iso(dt),
                    temporal_type="absolute",
                    char_start=m.start(),
                    char_end=m.end(),
                ))
            except ValueError:
                pass

        # 4. Relative: "N units ago" / "N units prior"
        for pattern in [_RELATIVE_AGO_RE, _RELATIVE_PRIOR_RE]:
            for m in pattern.finditer(text):
                n = int(m.group(1))
                days = _to_unit_days(n, m.group(2))
                approx_date = self.reference_date - timedelta(days=days)
                expressions.append(TemporalExpression(
                    raw_text=m.group(),
                    normalized_date=_normalize_to_iso(approx_date),
                    temporal_type="relative",
                    char_start=m.start(),
                    char_end=m.end(),
                    context=f"~{days} days before reference",
                ))

        # 5. Post-operative day
        for m in _POST_OP_RE.finditer(text):
            day_n = int(m.group(1))
            expressions.append(TemporalExpression(
                raw_text=m.group(),
                normalized_date=None,
                temporal_type="relative",
                char_start=m.start(),
                char_end=m.end(),
                context=f"post-op day {day_n}",
            ))

        # 6. Named anchors
        for m in _NAMED_ANCHOR_RE.finditer(text):
            expressions.append(TemporalExpression(
                raw_text=m.group(),
                normalized_date=m.group().lower(),  # keep as semantic string
                temporal_type="anchor",
                char_start=m.start(),
                char_end=m.end(),
            ))

        # 7. Durations (for X weeks / over the past Y months)
        for m in _DURATION_RE.finditer(text):
            n = int(m.group(1))
            days = _to_unit_days(n, m.group(2))
            start_date = self.reference_date - timedelta(days=days)
            expressions.append(TemporalExpression(
                raw_text=m.group(),
                normalized_date=_normalize_to_iso(start_date),
                temporal_type="duration",
                char_start=m.start(),
                char_end=m.end(),
                context=f"duration ~{days} days",
            ))

        # Sort by position and de-duplicate overlapping spans
        expressions.sort(key=lambda x: x.char_start)
        return self._deduplicate(expressions)

    @staticmethod
    def _deduplicate(expressions: List[TemporalExpression]) -> List[TemporalExpression]:
        """Remove expressions that are fully contained within a longer one."""
        result = []
        for expr in expressions:
            if result and expr.char_start < result[-1].char_end:
                # Prefer the longer span
                if (expr.char_end - expr.char_start) > (result[-1].char_end - result[-1].char_start):
                    result[-1] = expr
            else:
                result.append(expr)
        return result

    def associate_entity_to_time(
        self,
        text: str,
        entity: str,
        expressions: List[TemporalExpression],
        window_chars: int = 120,
    ) -> Optional[TemporalExpression]:
        """
        Find the temporal expression closest to `entity` in the surrounding text window.
        Returns the closest expression within `window_chars` characters, or None.
        """
        m = re.search(re.escape(entity), text, flags=re.IGNORECASE)
        if not m:
            return None
        entity_center = (m.start() + m.end()) // 2

        best: Optional[TemporalExpression] = None
        best_dist = window_chars + 1

        for expr in expressions:
            expr_center = (expr.char_start + expr.char_end) // 2
            dist = abs(entity_center - expr_center)
            if dist < best_dist:
                best_dist = dist
                best = expr

        return best if best_dist <= window_chars else None


# ---------------------------------------------------------------------------
# High-Level Pipeline Function
# ---------------------------------------------------------------------------

def build_temporal_timeline(
    text: str,
    extracted_data: dict,
    reference_date: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Extract temporal expressions from text and associate them with clinical entities.

    Returns:
    {
      "expressions": [ {raw_text, normalized_date, temporal_type, ...}, ... ],
      "entity_times": {
        "diagnoses": { "Pneumonia": "2024-01-12", ... },
        "procedures": { "Appendectomy": "on admission", ... },
        "medications": { "Aspirin": null, ... },
      }
    }
    """
    extractor = TemporalExtractor(reference_date=reference_date)
    expressions = extractor.extract(text)

    entity_times: Dict[str, Dict[str, Optional[str]]] = {
        "diagnoses": {},
        "medications": {},
        "procedures": {},
    }

    for diag in extracted_data.get("diagnoses", []):
        assoc = extractor.associate_entity_to_time(text, diag, expressions)
        entity_times["diagnoses"][diag] = assoc.normalized_date if assoc else None

    for med in extracted_data.get("medications", []):
        name = med.get("name", "") if isinstance(med, dict) else str(med)
        if name:
            assoc = extractor.associate_entity_to_time(text, name, expressions)
            entity_times["medications"][name] = assoc.normalized_date if assoc else None

    for proc in extracted_data.get("procedures", []):
        assoc = extractor.associate_entity_to_time(text, proc, expressions)
        entity_times["procedures"][proc] = assoc.normalized_date if assoc else None

    return {
        "expressions": [e.to_dict() for e in expressions],
        "entity_times": entity_times,
    }
