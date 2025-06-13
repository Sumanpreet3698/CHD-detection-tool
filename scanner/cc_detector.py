# cc_detector.py
import spacy
try:
    _nlp = spacy.load("en_core_web_lg")
except Exception as e:
    print(f"[WARN] Failed to load spaCy model in cc_detector: {e}")

from presidio_analyzer import AnalyzerEngine

# Default engine: Presidio built-in recognizers
engine_default = AnalyzerEngine()

def detect_credit_cards_default(text: str, score_threshold: float = None):
    """
    Use Presidio's built-in CreditCardRecognizer (Luhn + context).
    """
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


# # 2) Custom engine: copy a separate AnalyzerEngine and register our custom PatternRecognizer(s)
# engine_custom = AnalyzerEngine()

# # Define custom regex patterns if needed (16-digit with spaces/hyphens, etc.)
# full_cc_pattern = Pattern(
#     name="custom_full_cc",
#     # Matches 16-digit sequences with optional spaces or hyphens
#     regex=r"\b(?:\d{4}[- ]?){3}\d{4}\b",
#     score=0.5,
# )

# custom_recognizer = PatternRecognizer(
#     supported_entity="CREDIT_CARD",
#     patterns=[full_cc_pattern],
#     name="CustomCCRecognizer"
# )
# engine_custom.registry.add_recognizer(custom_recognizer)


# def detect_credit_cards_custom(text: str, score_threshold: float = 0.1):
#     """
#     Use our custom recognizer for broader regex matching.
#     Returns list of Presidio Result objects.
#     """
#     # Lower threshold by default to catch more; caller may override.
#     return engine_custom.analyze(
#         text=text,
#         entities=["CREDIT_CARD"],
#         language="en",
#         score_threshold=score_threshold,
#     )
