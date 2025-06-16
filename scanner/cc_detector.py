# cc_detector.py
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, RecognizerRegistry
import spacy

# Load spaCy model for context-aware detection
try:
    nlp = spacy.load("en_core_web_lg")
except Exception as e:
    print(f"[WARN] Failed to load spaCy model in cc_detector: {e}")
    nlp = None

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

# Context keywords & doc
CONTEXT_KEYWORDS = [
    "credit", "card", "visa", "mastercard", "payment", "debit", "billing", "invoice",
    "transaction", "purchase", "checkout", "order", "receipt", "cvv", "cvc", "expiry",
    "expiration", "valid", "account", "bank", "charge", "statement", "online", "atm", "pin", "secure"
]

if nlp:
    context_doc = nlp(" ".join(CONTEXT_KEYWORDS))
else:
    context_doc = None


# 2) Custom context-aware recognizer, uses weight scoring to combine context similarity, pattern matching, and Luhn algorithm.
def detect_credit_cards_custom(
    text: str,
    base_threshold: float = 0.1,
    final_threshold: float = 0.5,
    w_context: float = 0.7,
    w_pattern: float = 0.3,
    window_size: int = 100
):
    """
    Returns list of dicts for detected credit-card matches with contextual scoring.
    """
    results = []
    if not text or not text.strip():
        return results

    # 1) Presidio built-in detection (regex+Luhn+minimal context) with low base threshold
    try:
        pres_res = engine_default.analyze(
            text=text,
            entities=["CREDIT_CARD"],
            language="en",
            score_threshold=base_threshold,
        )
    except Exception as e:
        print(f"[ERROR] Presidio analyze error: {e}")
        return results

    use_context = (nlp is not None and context_doc is not None)
    for r in pres_res:
        start, end = r.start, r.end
        match_text = text[start:end]
        base_score = r.score

        # 2) Compute context similarity if possible
        context_score = None
        if use_context:
            win_start = max(0, start - window_size)
            win_end = min(len(text), end + window_size)
            window_text = text[win_start:win_end].strip()
            if window_text:
                try:
                    window_doc = nlp(window_text)
                    sim = context_doc.similarity(window_doc)
                    # clamp to [0,1]
                    context_score = max(0.0, min(1.0, sim))
                except Exception:
                    context_score = 0.0
            else:
                context_score = 0.0

        # 3) Combine into final_score
        if context_score is None:
            final_score = base_score
        else:
            final_score = w_context * context_score + w_pattern * base_score
            final_score = max(0.0, min(1.0, final_score))

        # 4) Filter by final_threshold
        if final_score >= final_threshold:
            results.append({
                "start": start,
                "end": end,
                "entity_type": r.entity_type,
                "match": match_text,
                "base_score": base_score,
                "context_score": context_score,
                "final_score": final_score
            })
    return results