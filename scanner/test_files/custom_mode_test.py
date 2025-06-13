# quick_test.py
from cc_detector import detect_credit_cards_custom, detect_credit_cards_default
from text_extractor import extract_text_from_pdf  # example extractor

# load a known test file
path = r"C:\Users\nikit\OneDrive - Panacea InfoSec Pvt. Ltd\Panacea_Infosec\forensic_scanner\cc_scanner\sample_docs\pdf_test2.pdf"
for label, text in extract_text_from_pdf(path):
    # Test default
    res_def = detect_credit_cards_default(text, score_threshold=0.1)
    print("Default results:", [text[r.start:r.end] for r in res_def])
    # Test custom
    res_cust = detect_credit_cards_custom(text, score_threshold=0.1)
    print("Custom results:", [text[r.start:r.end] for r in res_cust])
