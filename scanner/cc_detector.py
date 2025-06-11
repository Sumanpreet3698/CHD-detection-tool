from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from .luhn import is_luhn_valid

engine = AnalyzerEngine()

# Full 16-digit CC pattern (spaces or hyphens allowed)
full_cc = Pattern(
    name="full_cc",
    regex=(
        r"\b(?:4[0-9]{3}(?:[ -]?[0-9]{4}){3}"
        r"|5[1-5][0-9]{2}(?:[ -]?[0-9]{4}){3}"
        r"|3[47][0-9]{2}(?:[ -]?[0-9]{4}){2}"
        r"|6(?:011|5[0-9]{2})(?:[ -]?[0-9]{4}){3})\b"
    ),
    score=0.5,
)

engine.registry.add_recognizer(
    PatternRecognizer(
        supported_entity="CREDIT_CARD",
        patterns=[full_cc],
        name="FullCC"
    )
)

def detect_credit_cards(text: str):
    """
    Run Presidio detection + Luhn check.
    Returns only Luhn-valid matches.
    """
    results = engine.analyze(text=text, entities=["CREDIT_CARD"], language="en")
    valid = []
    for r in results:
        candidate = text[r.start:r.end]
        candidate_clean = candidate.replace(" ", "").replace("-", "")
        if is_luhn_valid(candidate_clean):
            valid.append(r)
    return valid
