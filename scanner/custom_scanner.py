#!/usr/bin/env python3
"""
custom_scanner.py

Scan a directory tree for files potentially containing credit-card numbers,
using detect_credit_cards_custom() from cc_detector.py which combines Presidio
built-in detection (regex+Luhn) with spaCy-based context similarity scoring.

Outputs a CSV report with columns:
 timestamp, file, label, start, end, match_text, base_score, context_score, final_score

CLI options allow tuning:
 --base-threshold: Presidio base_score threshold (low: e.g. 0.1 to capture all Luhn-valid candidates)
 --final-threshold: final_score cutoff after combining context and base (e.g. 0.5)
 --w-context: weight for context_score in final_score
 --w-pattern: weight for base_score in final_score
 --window-size: number of characters around match to use for context similarity

Also supports checkpointing, exclusion of extensions, batch size adjustment, memory monitoring, multiprocessing, and quick prefilter by raw bytes.
"""

import os
import sys
import csv
import time
import pickle
import gc
import psutil
import logging
import re
import argparse
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Import your modules; ensure PYTHONPATH includes the directory containing these files.
from config import EXTRACTORS
from cc_detector import (
    detect_credit_cards_custom,
    detect_credit_cards_transformer,
)

# ------ Configuration defaults ------
CHECKPOINT = "checkpoint_custom_scan.pkl"
REPORT = "scan_report_custom.csv"

# Worker/memory settings
total_mem_gb = psutil.virtual_memory().total / (1024**3)
# Estimate ~1GB per worker; adjust if needed
est_worker_mem_gb = 1.0
# Determine WORKERS dynamically but cap at cpu_count()-1
if cpu_count() > 1:
    WORKERS = 2 # max(1, min(int(total_mem_gb / est_worker_mem_gb) - 1, cpu_count() - 1))
else:
    WORKERS = 1
# Fallback if the above yields 0
if WORKERS < 1:
    WORKERS = 1

INITIAL_BATCH_SIZE = 500
MAXTASKSPERCHILD = 100
MEMORY_WARN_THRESHOLD = 0.8  # warn if system memory used > 80%
MAX_EXTRACT_SIZE = 100 * 1024 * 1024  # 100 MB: above this, do quick prefilter
MAX_TIME_PER_FILE = 60  # seconds per file max
# Quick prefilter regex (raw bytes) for any 16-digit-like pattern
PREFILTER_REGEX = re.compile(rb"\b(?:\d{4}[- ]?){3}\d{4}\b")
# Priority extensions to yield early in scanning
PRIORITY_EXTENSIONS = {".txt", ".csv", ".docx", ".pdf"}

# Logging setup
logging.basicConfig(
    filename="custom_scanner.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def gather_all_files_scandir(root, ext_set, exclude_exts=None):
    """
    Recursively yield all file paths under 'root' whose extension is in ext_set,
    using os.scandir for efficiency. If exclude_exts provided, skip those extensions.
    Yields in no particular order; prioritizes files whose ext in PRIORITY_EXTENSIONS
    by yielding them first (since directories are scanned in arbitrary order,
    the priority logic here is just to check ext membership early).
    """
    stack = [root]
    while stack:
        dirpath = stack.pop()
        try:
            with os.scandir(dirpath) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            _, ext = os.path.splitext(entry.name)
                            ext = ext.lower()
                            if exclude_exts and ext in exclude_exts:
                                continue
                            # Only include if in ext_set
                            if ext in PRIORITY_EXTENSIONS:
                                yield entry.path
                            elif ext in ext_set:
                                yield entry.path
                    except Exception:
                        continue
        except Exception:
            continue


def batched_file_iterator(gen, batch_size, done_set=None):
    """
    Given a generator 'gen' yielding file paths, collect into lists of size batch_size,
    skipping files in done_set if provided.
    """
    batch = []
    for path in gen:
        if done_set and path in done_set:
            continue
        batch.append(path)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def init_worker():
    """
    Initializer for worker processes.
    Importing cc_detector ensures spaCy/Presidio engine is loaded once per worker.
    """
    import cc_detector  # Preload cc_detector within each worker


def quick_scan_bytes(path, max_bytes=65536):
    """
    Quick prefilter: read up to max_bytes from start and end of file in binary,
    and search for any 16-digit-like pattern. Returns True if pattern found (i.e.
    file is candidate), False if none found (so skip heavy extraction).
    If any error reading, return True (to avoid false negatives).
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(min(max_bytes, size))
            data = head
            if size > max_bytes:
                try:
                    f.seek(-max_bytes, os.SEEK_END)
                    tail = f.read(min(max_bytes, size))
                    data += tail
                except Exception:
                    pass
        return bool(PREFILTER_REGEX.search(data))
    except Exception:
        return True


def process_file_wrapper(args):
    """
    Worker function for processing one file.
    args: tuple(path, base_threshold, final_threshold, w_context, w_pattern, window_size, engine_type, transformer_model)
    Returns: (path, hits_list), where hits_list is list of tuples:
      (path, label, start, end, match_text, base_score, context_score, final_score)
    """
    (
        path,
        base_threshold,
        final_threshold,
        w_context,
        w_pattern,
        window_size,
        engine_type,
        transformer_model,
    ) = args
    start_time = time.time()
    hits = []

    # Check file size
    try:
        size = os.path.getsize(path)
    except Exception:
        size = None

    _, ext = os.path.splitext(path)
    ext = ext.lower()
    extractor = EXTRACTORS.get(ext)
    if extractor is None:
        return (path, hits)

    # Prefilter large files by raw scan
    if size is not None and size > MAX_EXTRACT_SIZE:
        if not quick_scan_bytes(path):
            logger.info(f"Skipping large file (no CC-like pattern): {path} ({size} bytes)")
            return (path, hits)
        else:
            logger.warning(f"Large file passed quick-scan: processing {path} ({size} bytes)")

    # Extraction & detection
    try:
        for label, text in extractor(path):
            if not text or not text.strip():
                continue
            length = len(text)
            chunk_size = 50_000
            offset = 0
            while offset < length:
                # Timeout check
                if time.time() - start_time > MAX_TIME_PER_FILE:
                    logger.warning(f"Timeout processing file: {path}")
                    return (path, hits)
                end_off = min(length, offset + chunk_size)
                chunk = text[offset:end_off]
                offset = end_off
                if not chunk.strip():
                    continue

                # Choose detection logic based on engine_type
                if engine_type == "transformer":
                    raw_results = detect_credit_cards_transformer(
                        chunk,
                        score_threshold=final_threshold,  # reuse threshold
                        model_name=transformer_model,
                    )
                    # Convert AnalyzerResult objects to the tuple format expected downstream
                    detections = []
                    for r in raw_results:
                        detections.append(
                            {
                                "start": r.start,
                                "end": r.end,
                                "entity_type": r.entity_type,
                                "match": chunk[r.start : r.end],
                                "base_score": r.score,
                                "context_score": None,
                                "final_score": r.score,
                            }
                        )
                else:
                    detections = detect_credit_cards_custom(
                        chunk,
                        base_threshold=base_threshold,
                        final_threshold=final_threshold,
                        w_context=w_context,
                        w_pattern=w_pattern,
                        window_size=window_size,
                    )
                for det in detections:
                    # Adjust indices by offset
                    start_idx = det["start"] + (offset - len(chunk))
                    end_idx = det["end"] + (offset - len(chunk))
                    snippet = det["match"].strip().replace("\n", " ")
                    hits.append((
                        path,
                        label,
                        start_idx,
                        end_idx,
                        snippet,
                        det.get("base_score"),
                        det.get("context_score"),
                        det.get("final_score"),
                    ))
    except Exception as e:
        logger.error(f"Error extracting/detecting in {path}: {e}", exc_info=True)
        return (path, hits)

    return (path, hits)


def save_checkpoint(done_set, checkpoint_path):
    try:
        with open(checkpoint_path, "wb") as f:
            pickle.dump(done_set, f)
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    return set()


def monitor_memory_and_adjust(batch_size):
    mem = psutil.virtual_memory()
    used_frac = mem.used / mem.total
    if used_frac > MEMORY_WARN_THRESHOLD:
        logger.warning(f"High memory usage: {used_frac:.2%}.")
        new_size = max(100, batch_size // 2)
        if new_size < batch_size:
            logger.info(f"Reducing batch size from {batch_size} to {new_size} due to memory pressure")
        return new_size
    return batch_size


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Custom Credit Card Detection Scanner with context-aware weighted scoring"
    )
    parser.add_argument("scan_root", type=str, help="Root directory to scan")
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpointing (by default, progress is saved to resume)"
    )
    parser.add_argument(
        "--exclude", type=str, nargs="*", default=None,
        help="File extensions to exclude, e.g. .png .jpg"
    )
    parser.add_argument(
        "--base-threshold", type=float, default=0.1,
        help="Base threshold for Presidio built-in detection (low to capture Luhn-valid). Default=0.1"
    )
    parser.add_argument(
        "--final-threshold", type=float, default=0.5,
        help="Final combined threshold (context+pattern). Default=0.5"
    )
    parser.add_argument(
        "--w-context", type=float, default=0.7,
        help="Weight for context_score in final_score. Default=0.7"
    )
    parser.add_argument(
        "--w-pattern", type=float, default=0.3,
        help="Weight for base_score in final_score. Default=0.3"
    )
    parser.add_argument(
        "--window-size", type=int, default=100,
        help="Window size (chars) around match for context similarity. Default=100"
    )
    parser.add_argument(
        "--batch-size", type=int, default=INITIAL_BATCH_SIZE,
        help=f"Initial batch size of files to process in parallel. Default={INITIAL_BATCH_SIZE}"
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="Override number of worker processes (default auto-detected)"
    )
    parser.add_argument(
        "--max-time-per-file", type=int, default=MAX_TIME_PER_FILE,
        help=f"Max seconds per file before timing out. Default={MAX_TIME_PER_FILE}"
    )
    parser.add_argument(
        "--nlp-engine",
        choices=["spacy", "transformer"],
        default="spacy",
        help="Choose underlying NLP engine for Presidio (default: spacy)",
    )
    parser.add_argument(
        "--transformer-model",
        type=str,
        default="bert-base-cased",
        help="HuggingFace model identifier for transformer engine",
    )
    return parser.parse_args()


def main():
    start_time_total = time.time()
    args = parse_arguments()
    scan_root = args.scan_root
    use_checkpoint = not args.no_checkpoint
    exclude_exts = set(args.exclude) if args.exclude else set()
    base_threshold = args.base_threshold
    final_threshold = args.final_threshold
    w_context = args.w_context
    w_pattern = args.w_pattern
    window_size = args.window_size
    batch_size = args.batch_size
    max_time_per_file = args.max_time_per_file
    engine_type = args.nlp_engine
    transformer_model = args.transformer_model

    global MAX_TIME_PER_FILE
    MAX_TIME_PER_FILE = max_time_per_file

    # Override WORKERS if provided
    global WORKERS
    if args.max_workers is not None and args.max_workers > 0:
        WORKERS = args.max_workers

    logger.info(f"Starting custom scan on: {scan_root}")
    logger.info(f"Exclude extensions: {exclude_exts}")
    logger.info(f"base_threshold={base_threshold}, final_threshold={final_threshold}, "
                f"w_context={w_context}, w_pattern={w_pattern}, window_size={window_size}")
    logger.info(f"Workers={WORKERS}, initial batch_size={batch_size}, max_time_per_file={MAX_TIME_PER_FILE}")

    # Gather files
    extensions = set(EXTRACTORS.keys()) - exclude_exts
    file_gen = gather_all_files_scandir(scan_root, extensions, exclude_exts)

    # Load checkpoint
    checkpoint_path = CHECKPOINT
    done = set()
    if use_checkpoint:
        done = load_checkpoint(checkpoint_path)
        logger.info(f"Loaded checkpoint: {len(done)} files already done")

    # Prepare CSV report
    report_path = REPORT
    write_header = not os.path.exists(report_path)
    report_file = open(report_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(report_file)
    if write_header:
        writer.writerow([
            "timestamp", "file", "label", "start", "end", "match",
            "base_score", "context_score", "final_score"
        ])

    processed_count = 0
    batch_num = 0

    # Count total files for progress bar (this may take a moment)
    total_files = sum(1 for _ in gather_all_files_scandir(scan_root, extensions, exclude_exts))
    with tqdm(total=total_files, desc="Scanning files") as pbar:
        for batch in batched_file_iterator(file_gen, batch_size, done if use_checkpoint else None):
            batch_num += 1
            logger.info(f"Starting batch {batch_num}: {len(batch)} files; batch_size={batch_size}")
            # Monitor/adjust memory
            batch_size = monitor_memory_and_adjust(batch_size)

            # Build args for worker: (path, base_threshold, final_threshold, w_context, w_pattern, window_size, engine_type, transformer_model)
            args_iter = (
                (
                    path,
                    base_threshold,
                    final_threshold,
                    w_context,
                    w_pattern,
                    window_size,
                    engine_type,
                    transformer_model,
                )
                for path in batch
            )
            with Pool(processes=WORKERS, initializer=init_worker, maxtasksperchild=MAXTASKSPERCHILD) as pool:
                # Determine an appropriate chunksize
                chunksize = max(1, len(batch) // (WORKERS * 4)) if WORKERS > 0 else 1
                for path, hits in pool.imap_unordered(process_file_wrapper, args_iter, chunksize):
                    # Use local timezone-aware timestamp instead of UTC
                    timestamp = datetime.now().astimezone().isoformat()
                    # Write hits if any
                    for hit in hits:
                        # hit: (path, label, start, end, snippet, base_score, context_score, final_score)
                        file_path, label, start_idx, end_idx, snippet, base_s, ctx_s, final_s = hit
                        writer.writerow([
                            timestamp,
                            file_path,
                            label,
                            start_idx,
                            end_idx,
                            snippet,
                            f"{base_s:.3f}" if base_s is not None else "",
                            f"{ctx_s:.3f}" if ctx_s is not None else "",
                            f"{final_s:.3f}" if final_s is not None else "",
                        ])
                    processed_count += 1
                    pbar.update(1)

            gc.collect()
            logger.info(f"Completed batch {batch_num}. Total processed so far: {processed_count}")
            if use_checkpoint:
                done.update(batch)
                save_checkpoint(done, checkpoint_path)

    report_file.close()
    elapsed_total = time.time() - start_time_total
    logger.info(f"Custom scan completed. Elapsed time: {elapsed_total:.1f}s; processed {processed_count} files.")
    if use_checkpoint:
        logger.info(f"Final checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
