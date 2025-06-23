# Makes the `scanner` directory a Python package so that its modules
# can be imported via the dotted path (e.g. ``scanner.cc_detector``)
# from anywhere on the PYTHONPATH.

__all__ = [
    "cc_detector",
    "custom_scanner",
    "optimized_scanner",
    "text_extractor",
    "config",
] 