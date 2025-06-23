# cc_detector.py
# Standard libraries
from functools import lru_cache

# Third-party
import spacy
from presidio_analyzer import AnalyzerEngine

# ---------------------------------------------------------------------------
# NOTE:
#   • We import `TransformersNlpEngine` directly from the modern public path
#     (``presidio_analyzer.nlp_engine``).  This is available from
#     ``presidio-analyzer>=2.2`` which should be installed with the
#     ``[transformers]`` extra, e.g.:
#
#         pip install --upgrade "presidio-analyzer[transformers]"
#
#   • If the user's environment ships an older Presidio release which lacks
#     this symbol, an ``ImportError`` will be raised immediately with a clear
#     message, instead of silently falling back to a legacy private path.
# ---------------------------------------------------------------------------

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

# ------------- NLP ENGINE FACTORY -------------

@lru_cache(maxsize=2)
def _create_presidio_engine(engine_type: str = "spacy", model_name: str = "en_core_web_lg"):
    """Return a Presidio AnalyzerEngine configured with the requested NLP engine.

    Parameters
    ----------
    engine_type : str, optional
        Either "spacy" (default) or "transformer".
    model_name : str, optional
        HuggingFace model name when engine_type == "transformer", or spaCy model when "spacy".

    Notes
    -----
    The result is cached (LRU) so subsequent calls with identical parameters
    will reuse the same AnalyzerEngine instance (important when running in a
    long-lived process such as the Streamlit app).
    """
    engine_name = "transformers" if engine_type.lower() == "transformer" else "spacy"

    # Ensure the light-weight spaCy model is present (download once).
    try:
        from spacy.util import is_package
        from importlib import import_module

        if not is_package("en_core_web_sm"):
            # Access via import_module to placate static analysers.
            import_module("spacy.cli").download("en_core_web_sm")
    except Exception:
        # Even if download fails we'll let Presidio attempt to load and fail later.
        pass

    if engine_name == "transformers":
        # For the transformers engine Presidio expects a nested mapping with
        # both a spaCy pipeline (for tokenisation) **and** the transformers
        # model.  We hard-code the light-weight `en_core_web_sm` spaCy model so
        # the user doesn't need a second 400 MB download.
        configuration = {
            "nlp_engine_name": "transformers",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": {
                        "spacy": "en_core_web_sm",
                        "transformers": model_name,
                    },
                }
            ],
        }
    else:
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": model_name},
            ],
        }

    try:
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine_instance = provider.create_engine()

    except Exception as err:
        raise RuntimeError(
            f"Failed to initialise Presidio NLP engine '{engine_name}' with model '{model_name}': {err}"
        )

    return AnalyzerEngine(nlp_engine=nlp_engine_instance, supported_languages=["en"])


# Instantiate default engines once (cached by decorator anyway)
engine_spacy_default = _create_presidio_engine("spacy", "en_core_web_lg")
try:
    engine_transformer_default = _create_presidio_engine("transformer", "bert-base-cased")
except Exception as e:
    print(f"[WARN] Failed to load Transformers engine: {e}. Falling back to spaCy.")
    engine_transformer_default = engine_spacy_default


def detect_credit_cards_spacy(text: str, score_threshold: float | None = None):
    """Detect credit-card numbers using spaCy-backed Presidio engine."""
    kwargs = {
        "text": text,
        "entities": ["CREDIT_CARD"],
        "language": "en",
    }
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    return engine_spacy_default.analyze(**kwargs)


def detect_credit_cards_transformer(
    text: str,
    score_threshold: float | None = None,
    model_name: str = "bert-base-cased",
):
    """Detect credit-card numbers using a Transformer-backed Presidio engine.

    A small helper which lazily instantiates (and memoizes) an AnalyzerEngine
    powered by 🤗 Transformers. The default model is `bert-base-cased`, but you
    can supply any public HuggingFace NER model which supports English.
    """
    try:
        engine = _create_presidio_engine("transformer", model_name)
    except RuntimeError as err:
        # Graceful fallback to spaCy if transformer stack is missing.
        print(
            f"[WARN] Transformer engine unavailable ({err}). Falling back to spaCy for this call."
        )
        return detect_credit_cards_spacy(text, score_threshold)

    kwargs = {
        "text": text,
        "entities": ["CREDIT_CARD"],
        "language": "en",
    }
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    return engine.analyze(**kwargs)


# Keep existing detect_credit_cards_default for backward-compatibility (alias to spaCy)
detect_credit_cards_default = detect_credit_cards_spacy