# app.py
import streamlit as st
import pandas as pd
import os
import time
import csv
import gc
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from optimized_scanner import (
    gather_all_files_scandir,
    process_file_wrapper,
    EXTRACTORS,
    init_worker,
    MAXTASKSPERCHILD,
    INITIAL_BATCH_SIZE,
    monitor_memory_and_adjust
)

REPORT = "scan_report_full.csv"
WORKERS = max(1, cpu_count() - 1)

st.set_page_config(page_title="Credit Card Scanner", layout="wide")
st.title("💳 Credit Card Data Detector")

# Input section
folder_path = st.text_input("📁 Enter folder path to scan:")
threshold = st.slider("Presidio confidence threshold", 0.0, 1.0, 0.6, step=0.05)

if st.button("🚀 Start Scan"):
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        st.error("❌ Please enter a valid directory path.")
    else:
        st.info("⏳ Scanning started. Please wait...")
        start_time_total = time.time()

        extensions = set(EXTRACTORS.keys())
        files = list(gather_all_files_scandir(folder_path, extensions))

        total_files = len(files)
        report_rows = []

        # Prepare CSV
        write_header = not os.path.exists(REPORT)
        report_file = open(REPORT, "a", newline="", encoding="utf-8")
        writer = csv.writer(report_file)
        if write_header:
            writer.writerow(["timestamp", "file", "label", "start", "end", "match"])

        with st.spinner(f"Scanning {total_files} files..."):
            batch_num = 0
            current_batch_size = INITIAL_BATCH_SIZE

            for i in range(0, total_files, current_batch_size):
                batch = files[i: i + current_batch_size]
                current_batch_size = monitor_memory_and_adjust(current_batch_size)
                args_iter = ((path, threshold) for path in batch)

                with Pool(processes=WORKERS, initializer=init_worker, maxtasksperchild=MAXTASKSPERCHILD) as pool:
                    for path, hits in pool.imap_unordered(process_file_wrapper, args_iter):
                        timestamp = datetime.now(timezone.utc).isoformat()
                        for hit in hits:
                            row = [timestamp, *hit]
                            writer.writerow(row)
                            report_rows.append({
                                "timestamp": timestamp,
                                "file": hit[0],
                                "label": hit[1],
                                "start": hit[2],
                                "end": hit[3],
                                "match": hit[4]
                            })

                gc.collect()
                batch_num += 1

        report_file.close()
        elapsed_total = time.time() - start_time_total

        if report_rows:
            df = pd.DataFrame(report_rows)
            st.success(f"✅ Scan complete in {elapsed_total:.1f} seconds. Found {len(df)} matches.")
            st.dataframe(df, use_container_width=True)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Report CSV", csv_data, file_name=REPORT, mime="text/csv")
        else:
            st.info("🎉 Scan complete. No potential credit card matches found.")
