# app_custom.py
import os
import time
import pandas as pd
import streamlit as st
import csv
from multiprocessing import Pool

from datetime import datetime, timezone
from custom_scanner import (
    gather_all_files_scandir,
    process_file_wrapper,
    EXTRACTORS,
    init_worker,
    MAXTASKSPERCHILD,
    monitor_memory_and_adjust,
    load_checkpoint,
    save_checkpoint
)

# Configuration
CHECKPOINT_FILE = "checkpoint_custom_scan.pkl"
REPORT_FILE = "scan_report_custom.csv"
# _cpu_count = os.cpu_count() or 1
WORKERS = 2 # max(1, _cpu_count - 1)

# Set up page
st.set_page_config(page_title="Advanced Credit Card Scanner", layout="wide")
st.title("🔍 Advanced Credit Card Data Detector")

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Scan Parameters")
    
    # Threshold settings
    base_threshold = st.slider(
        "Base threshold (Presidio)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.05,
        help="Minimum confidence for pattern matching (lower catches more potential matches)"
    )
    
    final_threshold = st.slider(
        "Final threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum combined confidence score to report"
    )
    
    col1, col2 = st.columns(2)
    w_context = col1.slider(
        "Context weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Weight for context similarity score"
    )
    w_pattern = col2.slider(
        "Pattern weight", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.05,
        help="Weight for pattern matching score"
    )
    
    window_size = st.number_input(
        "Context window size (chars)",
        min_value=10,
        max_value=500,
        value=100,
        help="Characters around match to consider for context"
    )
    
    # Model selection
    engine_type = st.radio(
        "NLP engine",
        options=["spacy", "transformer"],
        index=0,
        help="Underlying NLP engine used by Presidio for entity detection",
    )

    transformer_model = None
    if engine_type == "transformer":
        transformer_model = st.text_input(
            "Transformer model (🤗 HuggingFace)",
            value="bert-base-cased",
            help="Model identifier to load when using transformer engine",
        )

    # Advanced settings
    with st.expander("Advanced Settings"):
        batch_size = st.number_input(
            "Batch size",
            min_value=10,
            max_value=1000,
            value=500,
            help="Files processed per batch"
        )
        
        max_workers = st.number_input(
            "Max workers",
            min_value=1,
            max_value=os.cpu_count(),
            value=WORKERS,
            help="Parallel processes to use"
        )
        
        max_time_per_file = st.number_input(
            "Max seconds per file",
            min_value=1,
            max_value=300,
            value=60,
            help="Timeout for processing individual files"
        )
        
        use_checkpoint = st.checkbox(
            "Enable checkpointing",
            value=True,
            help="Save progress to resume interrupted scans"
        )

# Main input
folder_path = st.text_input("📁 Enter folder path to scan:", placeholder="C:/path/to/folder")

if st.button("🚀 Start Scan", type="primary"):
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        st.error("❌ Please enter a valid directory path.")
    else:
        # Initialize scanning
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_placeholder = st.empty()
        start_time_total = time.time()
        
        # Gather files
        extensions = set(EXTRACTORS.keys())
        files = list(gather_all_files_scandir(folder_path, extensions))
        total_files = len(files)
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total files to scan", total_files)
        col2.metric("Supported extensions", ", ".join(extensions))
        col3.metric("Workers", max_workers)
        
        # Initialize report
        write_header = not os.path.exists(REPORT_FILE)
        report_file = open(REPORT_FILE, "a", newline="", encoding="utf-8")
        writer = csv.writer(report_file)
        if write_header:
            writer.writerow([
                "timestamp", "file", "label", "start", "end", "match",
                "base_score", "context_score", "final_score"
            ])
        
        # Load checkpoint if enabled
        done = set()
        if use_checkpoint and os.path.exists(CHECKPOINT_FILE):
            done = load_checkpoint(CHECKPOINT_FILE)
            st.info(f"Resuming from checkpoint - {len(done)} files already processed")
        
        # Process files in batches
        processed_count = 0
        report_rows = []
        
        with st.spinner("Scanning in progress..."):
            for i in range(0, total_files, batch_size):
                batch = files[i:i + batch_size]
                
                # Skip already processed files if using checkpoint
                if use_checkpoint:
                    batch = [f for f in batch if f not in done]
                    if not batch:
                        continue
                
                # Update progress
                progress = min((i + len(batch)) / total_files, 1.0)
                progress_bar.progress(progress)
                elapsed = time.time() - start_time_total
                remaining = (elapsed / (i + 1)) * (total_files - i) if i > 0 else 0
                status_text.info(
                    f"Processed {i}/{total_files} files ({progress:.1%}) - "
                    f"Elapsed: {elapsed:.1f}s - Remaining: ~{remaining:.1f}s"
                )
                
                # Process batch
                args_iter = (
                    (
                        path,
                        base_threshold,
                        final_threshold,
                        w_context,
                        w_pattern,
                        window_size,
                        engine_type,
                        transformer_model or "bert-base-cased",
                    )
                    for path in batch
                )
                
                with Pool(processes=max_workers, initializer=init_worker, maxtasksperchild=MAXTASKSPERCHILD) as pool:
                    for path, hits in pool.imap_unordered(process_file_wrapper, args_iter):
                        timestamp = datetime.now(timezone.utc).isoformat()
                        for hit in hits:
                            # hit: (path, label, start, end, snippet, base_s, ctx_s, final_s)
                            row = [timestamp] + list(hit)
                            writer.writerow(row)
                            report_rows.append({
                                "timestamp": timestamp,
                                "file": hit[0],
                                "label": hit[1],
                                "start": hit[2],
                                "end": hit[3],
                                "match": hit[4],
                                "base_score": hit[5],
                                "context_score": hit[6],
                                "final_score": hit[7]
                            })
                        processed_count += 1
                
                # Update checkpoint
                if use_checkpoint:
                    done.update(batch)
                    save_checkpoint(done, CHECKPOINT_FILE)
                
                # Adjust batch size based on memory
                batch_size = monitor_memory_and_adjust(batch_size)
        
        report_file.close()
        elapsed_total = time.time() - start_time_total
        
        # Display results
        if report_rows:
            df = pd.DataFrame(report_rows)
            
            # Convert scores to numeric
            for col in ['base_score', 'context_score', 'final_score']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add severity column
            df['severity'] = df['final_score'].apply(
                lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.4 else 'Low'
            )
            
            # Show summary
            st.success(f"""
            ✅ Scan complete in {elapsed_total:.1f} seconds. 
            Found {len(df)} matches across {df['file'].nunique()} files.
            """)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total matches", len(df))
            col2.metric("Files with matches", df['file'].nunique())
            col3.metric("Avg. confidence", f"{df['final_score'].mean():.1%}")
            
            # Style function for highlighting
            def highlight_scores(row):
                styles = [''] * len(row)
                if row['final_score'] >= 0.7:
                    styles[-3:] = ['background-color: #4CAF50; color: white'] * 3  # Green for high confidence
                elif row['final_score'] < 0.4:
                    styles[-3:] = ['background-color: #F44336; color: white'] * 3  # Red for low confidence
                return styles
            
            # Display styled dataframe
            st.dataframe(
                df.style.apply(highlight_scores, axis=1),
                use_container_width=True,
                height=600
            )
            
            # Download button for full report only
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Full Report",
                data=csv_data,
                file_name="full_cc_report.csv",
                mime="text/csv"
            )
        else:
            st.info("🎉 Scan complete. No credit card matches found above the threshold.")

# Add clean-up option
if os.path.exists(CHECKPOINT_FILE):
    if st.sidebar.button("🧹 Clear Checkpoint"):
        os.remove(CHECKPOINT_FILE)
        st.sidebar.success("Checkpoint cleared")