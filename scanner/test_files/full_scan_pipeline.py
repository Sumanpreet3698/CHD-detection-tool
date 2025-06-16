# full_scan_pipeline.py
import os
import sys
import csv
import pickle
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from cc_detector import detect_credit_cards_default, detect_credit_cards_custom
from config import EXTRACTORS

# Configuration
CHECKPOINT = "checkpoint_full_scan.pkl"
REPORT = "scan_report_full.csv"
WORKERS = 2 #os.cpu_count() or 4

def gather_all_files(root):
    """
    Recursively gather all files under 'root' regardless of extension/size.
    Only include files whose extension is in EXTRACTORS; 
    but do not filter by file size.
    """
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in EXTRACTORS:
                fullpath = os.path.join(dirpath, fname)
                all_files.append(fullpath)
    return all_files

def process_file(path, mode: str):
    """
    Process one file: extract text via extractors, then detect CCNs via chosen mode.
    Returns list of hits: tuples (path, label, start, end, snippet).
    """
    ext = os.path.splitext(path)[1].lower()
    extractor = EXTRACTORS.get(ext)
    hits = []
    if extractor:
        # Extract all text segments
        for label, text in extractor(path):
            if not text:
                continue
            # For large text, chunk in windows (to limit memory), here chunk size 50k
            length = len(text)
            step = 50_000
            for i in range(0, length, step):
                chunk = text[i : i + step]
                if not chunk.strip():
                    continue
                # # Choose detection function
                # if mode == "default":
                results = detect_credit_cards_default(chunk)
                # else:
                #     # custom mode
                #     # Note: one may pass a threshold; here we use default in detect_credit_cards_custom
                #     results = detect_credit_cards_custom(chunk)
                for r in results:
                    start, end = i + r.start, i + r.end
                    snippet = chunk[r.start:r.end].strip().replace("\n", " ")
                    hits.append((path, label, start, end, snippet))
    return hits

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        try:
            with open(CHECKPOINT, "rb") as f:
                return pickle.load(f)
        except Exception:
            print("[WARN] Failed to load checkpoint; starting fresh.")
    return set()

def save_checkpoint(done_set):
    with open(CHECKPOINT, "wb") as f:
        pickle.dump(done_set, f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python full_scan_pipeline.py <scan_root> [--no-checkpoint]")
        # print("  mode: 'default' to use Presidio's built-in recognizer; 'custom' to use regex-based recognizer")
        sys.exit(1)
    scan_root = sys.argv[1]
    # mode = sys.argv[2].lower()
    use_checkpoint = True
    if "--no-checkpoint" in sys.argv:
        use_checkpoint = False

    # if mode not in ("default", "custom"):
    #     print("Invalid mode. Use 'default' or 'custom'.")
    #     sys.exit(1)

    print(f"[INFO] Starting full-scan on: {scan_root}")
    # print(f"[INFO] Mode: {mode} (\"default\": Presidio built-in; \"custom\": regex recognizer)")
    start_time = time.perf_counter()

    # Gather files
    all_files = gather_all_files(scan_root)
    total = len(all_files)
    print(f"[INFO] Total files to process: {total}")

    # Checkpoint: which files already done
    done = set()
    if use_checkpoint:
        done = load_checkpoint()
        print(f"[INFO] Loaded checkpoint: {len(done)} files already processed")

    to_process = [f for f in all_files if f not in done]
    print(f"[INFO] Files to process in this run: {len(to_process)}")

    # Prepare CSV report
    # mode_suffix = mode
    # report_path = REPORT.replace(".csv", f"_{mode_suffix}.csv")
    write_header = not os.path.exists(REPORT)
    report_file = open(REPORT, "a", newline="", encoding="utf-8")
    writer = csv.writer(report_file)
    if write_header:
        writer.writerow(["timestamp","file","label","start","end","match"])

    # Parallel execution
    processed_count = 0
    with ProcessPoolExecutor(WORKERS) as executor:
        future_to_path = {executor.submit(process_file, path): path for path in to_process}
        for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Scanning"):
            path = future_to_path[future]
            try:
                hits = future.result()
                timestamp = datetime.utcnow().isoformat()
                for hit in hits:
                    # hit is (path, label, start, end, snippet)
                    writer.writerow([timestamp, *hit])
                processed_count += 1
                # Update checkpoint
                if use_checkpoint:
                    done.add(path)
                    # Periodically save checkpoint every 100 files
                    if processed_count % 100 == 0:
                        save_checkpoint(done)
            except Exception as e:
                print(f"[ERROR] {path}: {e}", file=sys.stderr)

    # Final checkpoint save
    if use_checkpoint:
        save_checkpoint(done)
    report_file.close()

    elapsed = time.perf_counter() - start_time
    print(f"[INFO] Full scan completed. Elapsed time: {elapsed:.1f}s for {total} files.")
    print(f"[INFO] Report written to: {REPORT}")
    if use_checkpoint:
        print(f"[INFO] Final checkpoint saved to: {CHECKPOINT}")

if __name__ == "__main__":
    main()
