import os, sys, csv, pickle, time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from . import quick_scan
from .cc_detector import detect_credit_cards
from . import config

WHITELIST = set(config.EXTRACTORS.keys())
MAX_SIZE = 500 * 1024 * 1024
CHECKPOINT = "checkpoint.pkl"
REPORT = "scan_report.csv"
WORKERS = os.cpu_count() or 4

# File types appropriate for binary-level quick scan
TEXT_LIKE = {".txt", ".log", ".csv", ".ini", ".cfg", ".bak", ".tmp"}

def gather_files(root):
    """
    Recursively collect files using os.scandir (faster than os.walk).
    """
    stack = [root]
    files = []
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in WHITELIST:
                            try:
                                if entry.stat().st_size <= MAX_SIZE:
                                    files.append(entry.path)
                            except OSError:
                                continue
        except OSError:
            continue
    return files

def process_file(path):
    ext = os.path.splitext(path)[1].lower()

    # Only run binary quick scan for plain text-like files
    if ext in TEXT_LIKE and not quick_scan.quick_cc_scan(path):
        return []

    extractor = config.EXTRACTORS.get(ext)
    hits = []
    if extractor:
        for label, text in extractor(path):
            for i in range(0, len(text), 50_000):
                chunk = text[i : i + 50_000]
                for r in detect_credit_cards(chunk):
                    start, end = i + r.start, i + r.end
                    snippet = chunk[r.start:r.end].strip().replace("\n", " ")
                    hits.append((path, label, start, end, snippet))
    return hits

def load_checkpoint():
    try:
        if os.path.exists(CHECKPOINT):
            with open(CHECKPOINT, "rb") as f:
                return pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        print("[WARN] Corrupt checkpoint file. Starting fresh.")
    return set()

def save_checkpoint(done):
    pickle.dump(done, open(CHECKPOINT, "wb"))

def main(root):
    print(f"[INFO] Scanning path: {root}")
    all_files = gather_files(root)
    print(f"[INFO] Total eligible files found: {len(all_files)}")

    done = load_checkpoint()
    to_process = [f for f in all_files if f not in done]

    print(f"[INFO] Skipping {len(done)} already processed files.")
    print(f"[INFO] Processing {len(to_process)} new files...")

    mode = "a" if os.path.exists(REPORT) else "w"
    with open(REPORT, mode, newline="", encoding="utf-8") as rep:
        writer = csv.writer(rep)
        if mode == "w":
            writer.writerow(["timestamp", "file", "label", "start", "end", "match"])

        start = time.perf_counter()
        with ProcessPoolExecutor(WORKERS) as exe:
            futures = {exe.submit(process_file, p): p for p in to_process}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning"):
                path = futures[fut]
                try:
                    for hit in fut.result():
                        timestamp = datetime.utcnow().isoformat()
                        writer.writerow([timestamp, *hit])
                    done.add(path)
                    if len(done) % 1000 == 0:
                        save_checkpoint(done)
                except Exception as e:
                    print(f"Error {path}: {e}", file=sys.stderr)

    save_checkpoint(done)
    elapsed = time.perf_counter() - start
    print(f"✅ Scan complete in {elapsed:.1f}s. Report saved to: {REPORT}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scanner.pipeline <scan_root>")
        sys.exit(1)
    main(sys.argv[1])
