"""
Cotton Boll Growth Detection - Auto Supervisor & Precision-Targeted Loop
========================================================================
Ensures training runs continuously until:
1. Target Precision/Accuracy of 87% - 90% is achieved, OR
2. All 100+ epochs are fully trained.

If the process ever pauses or exits prematurely, it automatically
cleans the memory and resumes immediately from last.pt.
"""

import os
import sys
import time
import subprocess
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "runs", "detect")
EXPERIMENT_NAME = "cotton_boll_detection"
LAST_PT = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "weights", "last.pt")
BEST_PT = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "weights", "best.pt")
RESULTS_CSV = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "results.csv")

TARGET_PRECISION = 0.87  # 87% Target Precision
TARGET_EPOCHS = 100


def get_metrics_summary():
    """Reads latest epoch and highest precision / mAP50 achieved."""
    current_epoch = 0
    best_precision = 0.0
    best_map50 = 0.0
    latest_precision = 0.0

    if os.path.exists(RESULTS_CSV):
        try:
            with open(RESULTS_CSV, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if len(lines) > 1:
                    headers = [h.strip() for h in lines[0].split(",")]
                    prec_idx = headers.index("metrics/precision(B)") if "metrics/precision(B)" in headers else 5
                    map_idx = headers.index("metrics/mAP50(B)") if "metrics/mAP50(B)" in headers else 7

                    for row in lines[1:]:
                        cols = [c.strip() for c in row.split(",")]
                        try:
                            ep = int(cols[0])
                            p = float(cols[prec_idx])
                            m = float(cols[map_idx])
                            if ep > current_epoch:
                                current_epoch = ep
                            if p > best_precision:
                                best_precision = p
                            if m > best_map50:
                                best_map50 = m
                            latest_precision = p
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass

    if current_epoch == 0 and os.path.exists(LAST_PT):
        try:
            ckpt = torch.load(LAST_PT, map_location="cpu", weights_only=False)
            current_epoch = int(ckpt.get("epoch", 0)) + 1
        except Exception:
            pass

    return {
        "epoch": current_epoch,
        "best_precision": best_precision,
        "latest_precision": latest_precision,
        "best_map50": best_map50,
    }


def is_training_running():
    """Checks if a python training process is currently running."""
    try:
        output = subprocess.check_output(
            ["pgrep", "-f", "python3 train.py"], text=True
        )
        pids = [p.strip() for p in output.strip().split("\n") if p.strip()]
        pids = [p for p in pids if int(p) != os.getpid()]
        return len(pids) > 0
    except subprocess.CalledProcessError:
        return False


def run_training_cycle():
    """Runs a training cycle with resume enabled."""
    env = os.environ.copy()
    env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    cmd = [sys.executable, "train.py", "--resume"]
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Launching: {' '.join(cmd)}")
    sys.stdout.flush()

    process = subprocess.Popen(
        cmd,
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()

    process.stdout.close()
    return process.wait()


def main():
    print("=" * 75)
    print("  Cotton Boll Detection - Precision-Targeted Auto Loop (Target: 87%-90%)")
    print("=" * 75)

    while True:
        metrics = get_metrics_summary()
        current_epoch = metrics["epoch"]
        best_p = metrics["best_precision"]
        best_m = metrics["best_map50"]

        print(
            f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] 📊 Epoch: {current_epoch}/{TARGET_EPOCHS} "
            f"| Best Precision: {best_p * 100:.1f}% (Goal: 87%-90%) "
            f"| Best mAP50: {best_m * 100:.1f}%"
        )
        sys.stdout.flush()

        # Check precision target threshold
        if best_p >= TARGET_PRECISION and current_epoch >= 50:
            print(f"🎉 GOAL ACHIEVED! Model reached {best_p * 100:.1f}% Precision (>= 87%)!")
            print(f"🏆 Best model weights ready at: {BEST_PT}")
            break

        if current_epoch >= TARGET_EPOCHS and best_p >= TARGET_PRECISION:
            print(f"🎉 TARGET REACHED: 100 Epochs Complete with {best_p * 100:.1f}% Precision!")
            break

        if is_training_running():
            print("⚡ Training process is actively running. Monitoring...")
            sys.stdout.flush()
            time.sleep(20)
            continue

        print(
            f"🔄 Continuing training toward 87%-90% target (Current Best: {best_p * 100:.1f}%)..."
        )
        sys.stdout.flush()
        exit_code = run_training_cycle()
        print(f"⚠️ Cycle ended (exit: {exit_code}). Re-launching in 5 seconds...")
        sys.stdout.flush()
        time.sleep(5)


if __name__ == "__main__":
    main()
