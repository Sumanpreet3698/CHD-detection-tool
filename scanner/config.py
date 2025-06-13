# config.py
import text_extractor as tx

TEXT_EXTS = {".txt", ".log", ".ini", ".cfg", ".bak", ".tmp"}

EXTRACTORS = {
    ".pdf": tx.extract_text_from_pdf,
    ".docx": tx.extract_text_from_docx,
    ".pptx": tx.extract_text_from_pptx,
    ".html": tx.extract_text_from_html,
    ".htm": tx.extract_text_from_html,
    ".xml": tx.extract_text_from_xml,
    ".xlsx": tx.extract_text_from_xlsx,
    ".xls": tx.extract_text_from_xlsx,
    ".csv": tx.extract_text_from_csv,
    **{ext: tx.extract_text_from_txt for ext in TEXT_EXTS},
    ".epub": tx.extract_text_from_epub,
    ".odt": tx.extract_text_from_odt,
    ".msg": tx.extract_text_from_msg,
    # ".png": tx.extract_text_from_image,
    # ".jpg": tx.extract_text_from_image,
    # ".jpeg": tx.extract_text_from_image,
    ".zip": tx.extract_text_from_zip,
    # ".jar": tx.extract_text_from_zip,
}
