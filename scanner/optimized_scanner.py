# optimized_scanner.py

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

from config import EXTRACTORS
from cc_detector import detect_credit_cards_default  # use default recognizer

# Configuration
CHECKPOINT = "checkpoint_full_scan.pkl"
REPORT = "scan_report_full.csv"

# Worker/memory settings (keep as before or adjust)
total_mem_gb = psutil.virtual_memory().total / (1024**3)
est_worker_mem_gb = 1.0
max_workers = 2  # or dynamic based on total_mem_gb
WORKERS = min(max_workers, cpu_count() - 1) if cpu_count() > 1 else 1
INITIAL_BATCH_SIZE = 500
MAXTASKSPERCHILD = 100
MEMORY_WARN_THRESHOLD = 0.8
MAX_EXTRACT_SIZE = 100 * 1024 * 1024  # 100MB
MAX_TIME_PER_FILE = 60  # seconds

PREFILTER_REGEX = re.compile(rb"\b(?:\d{4}[- ]?){3}\d{4}\b")
PRIORITY_EXTENSIONS = {".txt", ".csv", ".docx", ".pdf"}

logging.basicConfig(
    filename="pipeline.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def gather_all_files_scandir(root, ext_set):
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
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in PRIORITY_EXTENSIONS:
                                yield entry.path
                            elif ext in ext_set:
                                yield entry.path
                    except Exception:
                        continue
        except Exception:
            continue

def batched_file_iterator(gen, batch_size, done_set=None):
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
    import cc_detector  # preload spaCy/Presidio

def quick_scan_bytes(path, max_bytes=65536):
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
    args: tuple(path, threshold)
    Returns: (path, hits_list)
    """
    path, threshold = args
    start_time = time.time()
    hits = []
    try:
        size = os.path.getsize(path)
    except Exception:
        size = None

    ext = os.path.splitext(path)[1].lower()
    extractor = EXTRACTORS.get(ext)
    if not extractor:
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
        # Optional: debug logging to confirm extraction
        # segments = list(extractor(path))
        # if not segments: logger.info(f"No text segments from {path}")
        for label, text in extractor(path):
            if not text or not text.strip():
                continue
            length = len(text)
            chunk_size = 50_000
            for i in range(0, length, chunk_size):
                # Time check
                if time.time() - start_time > MAX_TIME_PER_FILE:
                    logger.warning(f"Timeout processing file: {path}")
                    return (path, hits)
                chunk = text[i: i + chunk_size]
                if not chunk.strip():
                    continue
                # Detect using default Presidio recognizer, with threshold
                results = detect_credit_cards_default(chunk, score_threshold=threshold)
                for r in results:
                    start_idx, end_idx = i + r.start, i + r.end
                    snippet = chunk[r.start:r.end].strip().replace("\n", " ")
                    hits.append((path, label, start_idx, end_idx, snippet))
    except Exception as e:
        logger.error(f"Error extracting {path}: {e}", exc_info=True)
        return (path, hits)

    return (path, hits)

def save_checkpoint(done_set):
    try:
        with open(CHECKPOINT, "wb") as f:
            pickle.dump(done_set, f)
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        try:
            with open(CHECKPOINT, "rb") as f:
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
    parser = argparse.ArgumentParser(description="Optimized Credit Card Detection Scanner")
    parser.add_argument("scan_root", type=str, help="Root directory to scan")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointing")
    parser.add_argument("--exclude", type=str, nargs="*", default=None,
                        help="File extensions to exclude (e.g., .png .jpg)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Presidio score threshold (0 to 1); default 0.6")
    return parser.parse_args()

def main():
    start_time_total = time.time()
    args = parse_arguments()
    scan_root = args.scan_root
    use_checkpoint = not args.no_checkpoint
    exclude_exts = set(args.exclude) if args.exclude else set()
    threshold = args.threshold

    logger.info(f"Starting optimized full-scan on: {scan_root}")
    logger.info(f"Exclude extensions: {exclude_exts}")
    logger.info(f"Presidio threshold: {threshold}")

    extensions = set(EXTRACTORS.keys()) - exclude_exts
    file_gen = gather_all_files_scandir(scan_root, extensions)

    done = set()
    if use_checkpoint:
        done = load_checkpoint()
        logger.info(f"Loaded checkpoint: {len(done)} files done")

    # Prepare CSV
    write_header = not os.path.exists(REPORT)
    report_file = open(REPORT, "a", newline="", encoding="utf-8")
    writer = csv.writer(report_file)
    if write_header:
        writer.writerow(["timestamp", "file", "label", "start", "end", "match"])

    processed_count = 0
    batch_num = 0
    current_batch_size = INITIAL_BATCH_SIZE

    # Count total files for progress bar
    total_files = sum(1 for _ in gather_all_files_scandir(scan_root, extensions))
    with tqdm(total=total_files, desc="Scanning files") as pbar:
        for batch in batched_file_iterator(file_gen, current_batch_size, done if use_checkpoint else None):
            batch_num += 1
            logger.info(f"Starting batch {batch_num}: {len(batch)} files; batch_size={current_batch_size}")
            current_batch_size = monitor_memory_and_adjust(current_batch_size)

            # Build args: each item is (path, threshold)
            args_iter = ((path, threshold) for path in batch)
            with Pool(processes=WORKERS, initializer=init_worker, maxtasksperchild=MAXTASKSPERCHILD) as pool:
                chunksize = max(1, len(batch) // (WORKERS * 4)) if WORKERS > 0 else 1
                for path, hits in pool.imap_unordered(process_file_wrapper, args_iter, chunksize):
                    # Write hits if any
                    timestamp = datetime.now(timezone.utc).isoformat()
                    for hit in hits:
                        writer.writerow([timestamp, *hit])
                    processed_count += 1
                    pbar.update(1)

            gc.collect()
            logger.info(f"Completed batch {batch_num}. Total processed: {processed_count}")
            if use_checkpoint:
                done.update(batch)
                save_checkpoint(done)

    report_file.close()
    elapsed_total = time.time() - start_time_total
    logger.info(f"Full scan completed. Elapsed time: {elapsed_total:.1f}s; processed {processed_count} files.")
    if use_checkpoint:
        logger.info(f"Final checkpoint saved to: {CHECKPOINT}")

if __name__ == "__main__":
    main()
