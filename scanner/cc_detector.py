# cc_detector.py
import spacy
try:
    _nlp = spacy.load("en_core_web_lg")
except Exception as e:
    print(f"[WARN] Failed to load spaCy model in cc_detector: {e}")

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry

# 1) Default engine: uses Presidio's built-in recognizers only.
engine_default = AnalyzerEngine()

def detect_credit_cards_default(text: str, score_threshold: float = None):
    if score_threshold is not None:
        return engine_default.analyze(
            text=text,
            entities=["CREDIT_CARD"],
            language="en",
            score_threshold=score_threshold,
        )
    else:
        return engine_default.analyze(
            text=text,
            entities=["CREDIT_CARD"],
            language="en",
        )

# 2) Custom context-aware recognizer: if you still want it, ensure threshold is set appropriately.
# Base regex pattern score:
cc_pattern = Pattern(
    name="cc_regex",
    regex=r"\b(?:\d{4}[- ]?){3}\d{4}\b",
    score=0.3  # base confidence
)
# Context words:
context_words = ["card", "credit", "visa", "mastercard", "payment", "debit", "number", "cc",
                 "expiry", "expiration", "valid", "cvv", "cvc", "billing", "account",
                 "bank", "transaction", "charge", "purchase", "checkout", "order", "receipt", "invoice"]
cc_recognizer = PatternRecognizer(
    supported_entity="CREDIT_CARD",
    patterns=[cc_pattern],
    context=context_words,
    name="ContextCC"
)
custom_registry = RecognizerRegistry()
custom_registry.add_recognizer(cc_recognizer)
custom_analyzer = AnalyzerEngine(registry=custom_registry)

def detect_credit_cards_custom(text: str, score_threshold: float = 0.3):
    """
    Custom analyzer: base score 0.3, so threshold must be <=0.3 to catch standalone matches.
    You can raise threshold if you want to require context boosting.
    """
    return custom_analyzer.analyze(
        text=text,
        entities=["CREDIT_CARD"],
        language="en",
        score_threshold=score_threshold,
    )
