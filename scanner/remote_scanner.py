#!/usr/bin/env python3
"""
remote_scanner.py

Scan a remote host's directory tree for credit-card data using the same
context-aware detection logic as the existing local scanner.

The script connects over SSH (via Paramiko), walks the remote directory with
SFTP, downloads candidate files to a temporary directory, and feeds them into
`process_file_wrapper` from `custom_scanner.py`. Results are written to a CSV
and also returned to the caller for programmatic use.
"""

from __future__ import annotations

import argparse
import csv
import os
import stat
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import List

import paramiko
from tqdm import tqdm

# Re-use existing components -------------------------------------------------
from config import EXTRACTORS
from custom_scanner import process_file_wrapper


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect_sftp(
    host: str,
    port: int,
    username: str,
    password: str | None = None,
    key_file: str | None = None,
):
    """Open an SSH + SFTP session and return the pair (ssh, sftp)."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        if key_file:
            pkey = paramiko.RSAKey.from_private_key_file(key_file)
            ssh.connect(hostname=host, port=port, username=username, pkey=pkey)
        else:
            ssh.connect(hostname=host, port=port, username=username, password=password)
    except Exception as exc:
        print(f"[!] SSH connection to {host}:{port} failed – {exc}", file=sys.stderr)
        raise

    return ssh, ssh.open_sftp()


def _remote_walk(sftp: paramiko.SFTPClient, root: str, extensions: set[str]):
    """Yield tuples (remote_path, size_in_bytes) for files with matching extensions."""
    stack: list[str] = [root.rstrip("/")]
    while stack:
        cur_dir = stack.pop()
        try:
            for entry in sftp.listdir_attr(cur_dir):
                name = entry.filename
                remote_path = f"{cur_dir}/{name}"
                mode = entry.st_mode or 0
                if stat.S_ISDIR(mode):
                    stack.append(remote_path)
                elif stat.S_ISREG(mode):
                    _, ext = os.path.splitext(name)
                    if ext.lower() in extensions:
                        size = entry.st_size if entry.st_size is not None else 0
                        yield remote_path, size
        except IOError:
            continue


def _download_temp(sftp: paramiko.SFTPClient, remote_path: str, tmp_dir: str) -> str:
    """Download *remote_path* to *tmp_dir*; return local filename."""
    basename = os.path.basename(remote_path)
    # Prefix with timestamp to avoid collisions.
    local_path = os.path.join(tmp_dir, f"{int(time.time() * 1000)}_{basename}")
    sftp.get(remote_path, local_path)
    return local_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remote_scan(
    *,
    host: str,
    port: int,
    username: str,
    password: str | None,
    key_file: str | None,
    remote_root: str,
    base_threshold: float = 0.1,
    final_threshold: float = 0.5,
    w_context: float = 0.7,
    w_pattern: float = 0.3,
    window_size: int = 100,
    engine_type: str = "spacy",
    transformer_model: str = "bert-base-cased",
    report_path: str = "scan_report_remote.csv",
    max_time_per_file: int = 60,
) -> tuple[List[dict], int]:
    """Scan *remote_root* on the remote host and return (detections, total_bytes)."""

    start_overall = time.time()

    # Honour per-file timeout if different from default
    import custom_scanner as _cs
    _cs.MAX_TIME_PER_FILE = max_time_per_file

    # 1) Connect ------------------------------------------------------------
    ssh, sftp = _connect_sftp(host, port, username, password, key_file)

    # 2) Enumerate remote files --------------------------------------------
    extensions = set(EXTRACTORS.keys())
    print("[+] Enumerating remote files…", flush=True)
    remote_items = list(_remote_walk(sftp, remote_root, extensions))
    total_files = len(remote_items)
    total_bytes = sum(size for _, size in remote_items)
    print(f"[+] Found {total_files} files ({total_bytes/1_048_576:.2f} MB) on remote host\n", flush=True)

    # 3) Prepare CSV output -------------------------------------------------
    write_header = not os.path.exists(report_path)
    report_fh = open(report_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(report_fh)
    if write_header:
        writer.writerow(
            [
                "timestamp",
                "file",
                "label",
                "start",
                "end",
                "match",
                "base_score",
                "context_score",
                "final_score",
            ]
        )

    # 4) Temporary workspace for downloads ---------------------------------
    detections: List[dict] = []
    processed_bytes = 0

    with tempfile.TemporaryDirectory(prefix="remote_scan_") as tmp_dir:
        for rpath, rsize in tqdm(remote_items, desc="Scanning remote files", unit="file"):
            try:
                local_path = _download_temp(sftp, rpath, tmp_dir)
            except Exception as exc:
                tqdm.write(f"[!] Failed to download {rpath}: {exc}")
                continue

            pw_args = (
                local_path,
                base_threshold,
                final_threshold,
                w_context,
                w_pattern,
                window_size,
                engine_type,
                transformer_model,
            )
            _, hits = process_file_wrapper(pw_args)

            # Convert hits into CSV rows / dicts, replacing path with *rpath*.
            for hit in hits:
                (
                    _local,
                    label,
                    start_idx,
                    end_idx,
                    snippet,
                    base_s,
                    ctx_s,
                    final_s,
                ) = hit

                row = [
                    datetime.now(timezone.utc).isoformat(),
                    rpath,
                    label,
                    start_idx,
                    end_idx,
                    snippet,
                    base_s,
                    ctx_s,
                    final_s,
                ]
                writer.writerow(row)
                detections.append(
                    {
                        "timestamp": row[0],
                        "file": rpath,
                        "label": label,
                        "start": start_idx,
                        "end": end_idx,
                        "match": snippet,
                        "base_score": base_s,
                        "context_score": ctx_s,
                        "final_score": final_s,
                    }
                )

            # Clean up temp file
            try:
                os.remove(local_path)
            except OSError:
                pass

            processed_bytes += rsize

    report_fh.close()
    ssh.close()

    elapsed = time.time() - start_overall
    print(f"\n[+] Remote scan complete in {elapsed:.1f}s – total matches: {len(detections)}")

    return detections, total_bytes


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="remote_scanner",
        description="Scan a remote host/directory tree for credit-card numbers via SSH/SFTP",
    )

    # Connection details ----------------------------------------------------
    p.add_argument("--host", required=True, help="Remote SSH hostname or IP")
    p.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    p.add_argument("--user", required=True, help="SSH username")

    auth = p.add_mutually_exclusive_group(required=False)
    auth.add_argument("--password", help="SSH password (omit when using key auth)")
    auth.add_argument("--key", help="Path to SSH private key for key-based auth")

    p.add_argument("--remote-path", required=True, help="Root directory on remote host to scan")

    # Detection tuning ------------------------------------------------------
    p.add_argument("--base-threshold", type=float, default=0.1)
    p.add_argument("--final-threshold", type=float, default=0.5)
    p.add_argument("--w-context", type=float, default=0.7)
    p.add_argument("--w-pattern", type=float, default=0.3)
    p.add_argument("--window-size", type=int, default=100)
    p.add_argument("--engine", choices=["spacy", "transformer"], default="spacy")
    p.add_argument("--transformer-model", default="bert-base-cased")

    # Misc ------------------------------------------------------------------
    p.add_argument("--report", default="scan_report_remote.csv", help="Output CSV path")
    p.add_argument("--max-time-per-file", type=int, default=60)

    return p


def main():
    cli = _build_cli()
    args = cli.parse_args()

    remote_scan(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        key_file=args.key,
        remote_root=args.remote_path,
        base_threshold=args.base_threshold,
        final_threshold=args.final_threshold,
        w_context=args.w_context,
        w_pattern=args.w_pattern,
        window_size=args.window_size,
        engine_type=args.engine,
        transformer_model=args.transformer_model,
        report_path=args.report,
        max_time_per_file=args.max_time_per_file,
    )


if __name__ == "__main__":
    main() 