"""
Cotton Boll Growth Detection - YOLOv8 Detection Training Script
================================================================
Trains a YOLOv8n model on the cotton boll growth detection dataset.
Uses Apple MPS (Metal Performance Shaders) for GPU acceleration on M2.

NOTE: Dataset has mixed detect/segment labels, so we use detection mode
      (yolov8n) which works with all annotations.
"""

import os
# Apple Silicon MPS memory optimizations
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from ultralytics import YOLO

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "data.yaml")
PROJECT_DIR = os.path.join(BASE_DIR, "runs", "detect")
EXPERIMENT_NAME = "cotton_boll_detection"

# Training hyperparameters
CONFIG = {
    # Model — detection model (not seg) since dataset has mixed labels
    "model": "yolov8n.pt",
    "data": DATA_YAML,
    "project": PROJECT_DIR,
    "name": EXPERIMENT_NAME,

    # Training schedule
    "epochs": 100,
    "patience": 20,                   # Early stopping patience
    "batch": 8,                       # Batch size (safe for M2 8GB+)
    "imgsz": 640,                     # Image size

    # Device
    "device": "mps",                  # Apple Silicon GPU

    # Optimizer
    "optimizer": "AdamW",
    "lr0": 0.001,                     # Initial learning rate
    "lrf": 0.01,                      # Final learning rate factor (cosine decay)
    "weight_decay": 0.0005,
    "warmup_epochs": 5,               # Warmup epochs
    "warmup_momentum": 0.8,

    # Augmentation (aggressive to help with class imbalance)
    "mosaic": 1.0,                    # Mosaic augmentation probability
    "mixup": 0.15,                    # Mixup augmentation
    "copy_paste": 0.1,               # Copy-paste augmentation (helps rare classes)
    "degrees": 10.0,                  # Rotation ±10°
    "translate": 0.2,                 # Translation ±20%
    "scale": 0.5,                     # Scale ±50%
    "fliplr": 0.5,                    # Horizontal flip
    "flipud": 0.1,                    # Vertical flip (minor for natural images)
    "hsv_h": 0.015,                   # HSV-Hue augmentation
    "hsv_s": 0.7,                     # HSV-Saturation augmentation
    "hsv_v": 0.4,                     # HSV-Value augmentation
    "erasing": 0.1,                   # Random erasing

    # Other
    "exist_ok": True,                 # Overwrite existing experiment
    "verbose": True,
    "plots": True,                    # Generate training plots
    "save": True,
    "save_period": 25,                # Save checkpoint every 25 epochs
    "val": True,
    "workers": 0,                     # 0 workers avoids macOS multiprocessing semaphore leaks
}


def main():
    print("=" * 70)
    print("  Cotton Boll Growth Detection - YOLOv8 Detection Training")
    print("=" * 70)
    print(f"\n📁 Dataset: {DATA_YAML}")
    print(f"📦 Model:   {CONFIG['model']}")
    print(f"🎯 Epochs:  {CONFIG['epochs']} (patience: {CONFIG['patience']})")
    print(f"🖼️  ImgSize: {CONFIG['imgsz']}")
    print(f"⚡ Device:  {CONFIG['device']}")
    print(f"📊 Batch:   {CONFIG['batch']}")
    print()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    last_model_path = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "weights", "last.pt")

    if args.resume and os.path.exists(last_model_path):
        print(f"🔄 Resuming training from: {last_model_path}")
        model = YOLO(last_model_path)
        results = model.train(resume=True)
    else:
        # Load pretrained model
        model = YOLO(CONFIG["model"])
        # Train
        results = model.train(**{k: v for k, v in CONFIG.items() if k != "model"})

    print("\n" + "=" * 70)
    print("  Training Complete!")
    print("=" * 70)

    # Validate on test set
    best_model_path = os.path.join(PROJECT_DIR, EXPERIMENT_NAME, "weights", "best.pt")
    print(f"\n🏆 Best model saved to: {best_model_path}")

    print("\n📊 Running validation on test set...")
    best_model = YOLO(best_model_path)
    test_results = best_model.val(
        data=DATA_YAML,
        split="test",
        device="mps",
        imgsz=640,
        plots=True,
        project=PROJECT_DIR,
        name=f"{EXPERIMENT_NAME}_test_eval",
        exist_ok=True,
    )

    print("\n✅ All done! Model is ready for deployment.")
    print(f"   Use the model at: {best_model_path}")
    print(f"   Launch the app:   streamlit run app.py")


if __name__ == "__main__":
    main()
