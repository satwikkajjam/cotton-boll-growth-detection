#!/usr/bin/env python3
"""
Training Live Monitor for Cotton Boll Growth Detection
======================================================
Usage:
    python3 monitor.py          # View current status snapshot
    python3 monitor.py --watch  # Live auto-refresh every 5 seconds
"""

import os
import sys
import time
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(BASE_DIR, "runs", "detect", "cotton_boll_detection", "results.csv")
WEIGHTS_DIR = os.path.join(BASE_DIR, "runs", "detect", "cotton_boll_detection", "weights")

def get_process_info():
    try:
        out = subprocess.check_output(
            "ps aux | grep -iE 'train\\.py|auto_runner\\.py' | grep -v grep | grep -v 'monitor\\.py'",
            shell=True,
            text=True
        ).strip()
        if not out:
            return []
        lines = []
        for l in out.splitlines():
            parts = l.split()
            if len(parts) > 10:
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                cmd = " ".join(parts[10:])
                lines.append(f"PID: {pid} (CPU: {cpu}%, MEM: {mem}%) -> {cmd}")
            else:
                lines.append(l)
        return lines
    except Exception:
        return []

def display_status():
    if os.isatty(sys.stdout.fileno()):
        os.system("clear" if os.name == "posix" else "cls")
    print("=" * 82)
    print(" 🌾  COTTON BOLL DETECTION - TRAINING MONITOR")
    print("=" * 82)

    # Process status
    processes = get_process_info()
    if processes:
        print("🟢 Training Process: ACTIVE")
        for p in processes:
            print(f"   {p}")
    else:
        print("🔴 Training Process: IDLE / NOT RUNNING")

    # Weights info
    if os.path.exists(WEIGHTS_DIR):
        last_pt = os.path.join(WEIGHTS_DIR, "last.pt")
        best_pt = os.path.join(WEIGHTS_DIR, "best.pt")
        if os.path.exists(last_pt):
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(last_pt)))
            print(f"💾 Last Saved Checkpoint : {mtime} (last.pt)")
        if os.path.exists(best_pt):
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(best_pt)))
            print(f"🏆 Best Checkpoint So Far: {mtime} (best.pt)")

    print("-" * 82)

    # Metrics
    if not os.path.exists(RESULTS_CSV):
        print("⚠️ results.csv not found yet.")
        return

    with open(RESULTS_CSV, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) <= 1:
        print("⚠️ Waiting for epoch 1 metrics to be written...")
        return

    headers = [h.strip() for h in lines[0].split(",")]
    rows = [r.split(",") for r in lines[1:]]

    def col_val(row, name, default=0.0):
        try:
            idx = headers.index(name)
            return float(row[idx].strip())
        except Exception:
            return default

    # Best statistics
    best_p = max(col_val(r, "metrics/precision(B)") for r in rows)
    best_m50 = max(col_val(r, "metrics/mAP50(B)") for r in rows)
    latest_row = rows[-1]
    curr_epoch = int(latest_row[0])

    print(f"📈 Progress: Epoch {curr_epoch}/100  |  Best Precision: {best_p*100:.1f}%  |  Best mAP50: {best_m50*100:.1f}%\n")
    print(f"{'Epoch':<7}{'Box Loss':<11}{'Cls Loss':<11}{'Precision':<13}{'Recall':<12}{'mAP50':<12}{'mAP50-95':<10}")
    print("-" * 82)

    # Show last 8 epochs
    for r in rows[-8:]:
        ep = int(r[0])
        box_loss = col_val(r, "train/box_loss")
        cls_loss = col_val(r, "train/cls_loss")
        prec = col_val(r, "metrics/precision(B)")
        rec = col_val(r, "metrics/recall(B)")
        map50 = col_val(r, "metrics/mAP50(B)")
        map50_95 = col_val(r, "metrics/mAP50-95(B)")
        print(f"{ep:<7}{box_loss:<11.4f}{cls_loss:<11.4f}{prec*100:<6.1f}%{'':<6}{rec*100:<6.1f}%{'':<5}{map50*100:<6.1f}%{'':<5}{map50_95*100:<6.1f}%")

    print("=" * 82)

def main():
    watch_mode = "--watch" in sys.argv
    if watch_mode:
        try:
            while True:
                display_status()
                print("Tip: Auto-refreshing every 5s. Press Ctrl+C to exit.")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped monitoring.")
    else:
        display_status()
        print("Tip: Run `python3 monitor.py --watch` to keep this updating live.")

if __name__ == "__main__":
    main()
