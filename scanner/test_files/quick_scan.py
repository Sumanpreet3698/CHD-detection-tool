import os
import regex as re

# Fast binary-level regex for full credit card patterns only
FAST_CC_RE = re.compile(
    br"\b(?:"                               # Only full cards
    br"4[0-9]{3}(?:[ -]?[0-9]{4}){3}|"      # Visa
    br"5[1-5][0-9]{2}(?:[ -]?[0-9]{4}){3}|" # MasterCard
    br"3[47][0-9]{2}(?:[ -]?[0-9]{4}){2}|"  # AMEX
    br"6(?:011|5[0-9]{2})(?:[ -]?[0-9]{4}){3}"  # Discover
    br")\b"
)

def quick_cc_scan(path, head=65536, tail=65536):
    """
    Read up to head bytes from start and tail bytes from end.
    Apply regex. Return True if any full CC match.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            data = f.read(head)
            if size > head:
                f.seek(max(size - tail, head))
                data += f.read(tail)
        return bool(FAST_CC_RE.search(data))
    except Exception:
        return False
