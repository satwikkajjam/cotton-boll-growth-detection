"""
Cotton Boll Growth Detection - Streamlit Web Application
=========================================================
A polished web interface for detecting cotton boll growth stages using YOLOv8.
Supports image upload for detection of 5 cotton growth stages.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import time

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cotton Boll Growth Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0d9488 0%, #065f46 50%, #1a472a 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(13, 148, 136, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 70%);
        animation: shimmer 8s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(5%, 5%); }
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    .hero-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(145deg, #f0fdf4, #dcfce7);
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(13, 148, 136, 0.15);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #065f46 !important;
        line-height: 1;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #1f2937 !important;
        margin-top: 0.3rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Class badge */
    .class-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 4px;
        color: #0f172a !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .class-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }

    /* Detection result card */
    .detection-card {
        background: #ffffff !important;
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        transition: all 0.2s ease;
    }
    .detection-card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    .detection-card strong {
        color: #0f172a !important;
    }

    /* ─── Sidebar (High Contrast & Sharp Readability) ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0fdf4 0%, #ecfdf5 100%) !important;
        border-right: 1px solid #d1fae5 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #064e3b !important;
        font-weight: 700 !important;
        margin-top: 0.5rem !important;
        letter-spacing: -0.3px !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span:not(.class-dot),
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        font-weight: 500 !important;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #064e3b !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"],
    [data-testid="stSidebar"] div[data-testid="stThumbValue"] {
        color: #064e3b !important;
        font-weight: 700 !important;
    }

    /* Expanders in Sidebar */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #a7f3d0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 6px rgba(6, 95, 70, 0.05) !important;
        margin-bottom: 0.6rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        padding: 8px 12px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        background: #f0fdf4 !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span {
        color: #064e3b !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
        fill: #064e3b !important;
        color: #064e3b !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: #ffffff !important;
        border-radius: 0 0 12px 12px !important;
        padding: 10px 14px 14px 14px !important;
        border-top: 1px solid #ecfdf5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] p {
        color: #1e293b !important;
        font-size: 0.88rem !important;
        line-height: 1.55 !important;
    }

    /* Header & Toolbar: KEEP visible so the sidebar reopen button works */
    header[data-testid="stHeader"],
    .stAppHeader {
        background: transparent !important;
    }
    
    [data-testid="stToolbar"],
    .stAppToolbar {
        background: transparent !important;
        visibility: visible !important;
        display: flex !important;
    }

    /* Hide ONLY Deploy button and 3-dots MainMenu */
    [data-testid="stAppDeployButton"],
    #MainMenu,
    [data-testid="stMainMenu"],
    footer {
        visibility: hidden !important;
        display: none !important;
    }

    /* ─── Sidebar Re-Open (Expand) Button (>>) ─── */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    button[aria-label="Expand sidebar"],
    button[aria-label="Open sidebar"] {
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: #ffffff !important;
        border: 2px solid #0d9488 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 15px rgba(13, 148, 136, 0.3) !important;
        color: #065f46 !important;
        cursor: pointer !important;
        padding: 4px 6px !important;
        transition: all 0.2s ease !important;
        z-index: 999999 !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stSidebarCollapsedControl"]:hover {
        background: #f0fdf4 !important;
        border-color: #065f46 !important;
        transform: scale(1.1) !important;
    }

    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapsedControl"] svg {
        fill: #065f46 !important;
        color: #065f46 !important;
        stroke: #065f46 !important;
    }

    /* Collapse button (<<) inside sidebar */
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        color: #064e3b !important;
    }
    [data-testid="stSidebarCollapseButton"] svg {
        fill: #064e3b !important;
    }

    /* Divider */
    .fancy-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #0d9488, transparent);
        margin: 1.5rem 0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Cotton Blossom",
    "Cotton Bud",
    "Early Boll",
    "Matured Cotton Boll",
    "Split Cotton Boll",
]

CLASS_COLORS = {
    "Cotton Blossom":      {"hex": "#f472b6", "rgb": (244, 114, 182), "bg": "#fdf2f8"},
    "Cotton Bud":          {"hex": "#34d399", "rgb": (52, 211, 153),  "bg": "#ecfdf5"},
    "Early Boll":          {"hex": "#60a5fa", "rgb": (96, 165, 250),  "bg": "#eff6ff"},
    "Matured Cotton Boll": {"hex": "#fbbf24", "rgb": (251, 191, 36), "bg": "#fffbeb"},
    "Split Cotton Boll":   {"hex": "#a78bfa", "rgb": (167, 139, 250),"bg": "#f5f3ff"},
}

GROWTH_INFO = {
    "Cotton Blossom": {
        "emoji": "🌸",
        "stage": "Flowering Stage",
        "description": "The cotton plant is actively flowering. White or cream-colored flowers appear, which will later develop into cotton bolls. This is a critical stage for pollination.",
    },
    "Cotton Bud": {
        "emoji": "🌱",
        "stage": "Budding Stage",
        "description": "Small, green buds (squares) are forming on the plant. These are the earliest visible signs of reproductive growth before flowering begins.",
    },
    "Early Boll": {
        "emoji": "🟢",
        "stage": "Early Boll Development",
        "description": "The flower has been fertilized and a small green boll is forming. The boll is compact and firm, with fibers developing inside the protective capsule.",
    },
    "Matured Cotton Boll": {
        "emoji": "🟤",
        "stage": "Mature Boll Stage",
        "description": "The boll has reached full size and is drying out. The outer shell is hardening and beginning to crack. The cotton fibers inside are fully developed.",
    },
    "Split Cotton Boll": {
        "emoji": "☁️",
        "stage": "Harvest Ready",
        "description": "The boll has split open, revealing fluffy white cotton fibers ready for harvest. This is the final growth stage indicating the crop is ready for picking.",
    },
}


# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained YOLOv8 model."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(base_dir, "models", "best_cotton_boll_model.pt"),
        os.path.join(base_dir, "runs", "detect", "cotton_boll_detection", "weights", "best.pt"),
        os.path.join(base_dir, "models", "best.pt"),
    ]

    model_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if not model_path:
        st.error(f"❌ Model not found. Checked:\n" + "\n".join(f"- `{p}`" for p in candidate_paths))
        st.stop()

    model = YOLO(model_path)
    return model



# ─── Helper Functions ────────────────────────────────────────────────────────
def run_inference(model, image, conf_threshold=0.25):
    """Run YOLOv8 inference on an image."""
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=0.45,
        imgsz=640,
        device="mps",
        verbose=False,
    )
    return results[0]


def draw_results(image, result):
    """Draw bounding boxes on the image with custom colors."""
    annotated = image.copy()

    if result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES[cls_id]
            color = CLASS_COLORS[cls_name]["rgb"]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw filled rectangle with transparency for the box area
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            annotated = cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0)

            # Draw box border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Draw label background
            label = f"{cls_name} {conf:.0%}"
            font_scale = 0.6
            thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Label background pill
            label_y = max(y1 - label_h - 14, label_h + 14)
            cv2.rectangle(annotated, (x1, label_y - label_h - 10), (x1 + label_w + 12, label_y + 4), color, -1)
            cv2.rectangle(annotated, (x1, label_y - label_h - 10), (x1 + label_w + 12, label_y + 4), color, 2)
            cv2.putText(annotated, label, (x1 + 6, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return annotated


def get_detection_summary(result):
    """Get a summary of detected objects."""
    summary = {}
    confidences = {}

    if result.boxes is not None and len(result.boxes) > 0:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES[cls_id]

            if cls_name not in summary:
                summary[cls_name] = 0
                confidences[cls_name] = []
            summary[cls_name] += 1
            confidences[cls_name].append(conf)

    return summary, confidences


# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1>🌿 Cotton Boll Growth Detection</h1>
        <p>AI-powered detection of cotton growth stages using YOLOv8</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        conf_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections. Lower = more detections (may include false positives).",
        )

        st.markdown("---")
        st.markdown("## 🏷️ Class Legend")

        for cls_name, colors in CLASS_COLORS.items():
            info = GROWTH_INFO[cls_name]
            st.markdown(
                f'<div class="class-badge" style="background: {colors["bg"]}; border: 1px solid {colors["hex"]};">'
                f'<span class="class-dot" style="background: {colors["hex"]};"></span>'
                f'{info["emoji"]} {cls_name}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("## 📖 Growth Stages")

        for cls_name, info in GROWTH_INFO.items():
            with st.expander(f"{info['emoji']} {info['stage']}"):
                st.write(info["description"])

    # Load model
    model = load_model()

    # Input section
    st.markdown("### 📸 Upload an Image")
    st.markdown("Upload a cotton plant image to detect growth stages.")

    col_upload, col_sample = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a photo of cotton plants to analyze their growth stage.",
            label_visibility="collapsed",
        )

    with col_sample:
        use_sample = st.button("🖼️ Use Sample Image", use_container_width=True)

    # Process image
    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    elif use_sample:
        # Load a sample image (sample_images for deployment, test/images as fallback)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_dir = os.path.join(base_dir, "sample_images")
        test_dir = os.path.join(base_dir, "test", "images")
        target_dir = sample_dir if (os.path.exists(sample_dir) and os.listdir(sample_dir)) else test_dir

        if os.path.exists(target_dir):
            sample_files = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if sample_files:
                import random
                sample_path = os.path.join(target_dir, random.choice(sample_files))
                image = cv2.imread(sample_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is not None:
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

        # Run inference with progress
        with st.spinner("🔍 Analyzing image..."):
            start_time = time.time()
            result = run_inference(model, image, conf_threshold)
            inference_time = time.time() - start_time

        # Draw results
        annotated_image = draw_results(image, result)
        summary, confidences = get_detection_summary(result)
        total_detections = sum(summary.values())

        # Stats row
        st.markdown("### 📊 Detection Results")
        stat_cols = st.columns(4)

        with stat_cols[0]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_detections}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_cols[1]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(summary)}</div>
                <div class="stat-label">Classes Found</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_cols[2]:
            avg_conf = np.mean([c for confs in confidences.values() for c in confs]) if confidences else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{avg_conf:.0%}</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with stat_cols[3]:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{inference_time:.2f}s</div>
                <div class="stat-label">Inference Time</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Image display
        col_orig, col_result = st.columns(2)

        with col_orig:
            st.markdown("#### 📷 Original Image")
            st.image(image, use_container_width=True)

        with col_result:
            st.markdown("#### 🎯 Detection Result")
            st.image(annotated_image, use_container_width=True)

        # Detailed results
        if summary:
            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🌱 Detected Growth Stages")

            for cls_name, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
                colors = CLASS_COLORS[cls_name]
                info = GROWTH_INFO[cls_name]
                avg_c = np.mean(confidences[cls_name])
                max_c = max(confidences[cls_name])

                st.markdown(
                    f'<div class="detection-card" style="border-left-color: {colors["hex"]};">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<div>'
                    f'<span style="font-size: 1.5rem;">{info["emoji"]}</span> '
                    f'<strong style="font-size: 1.1rem;">{cls_name}</strong>'
                    f'<span style="color: #9ca3af; margin-left: 8px;">({info["stage"]})</span>'
                    f'</div>'
                    f'<div style="text-align: right;">'
                    f'<span style="background: {colors["bg"]}; color: {colors["hex"]}; padding: 4px 12px; border-radius: 20px; font-weight: 700;">'
                    f'{count} detected</span>'
                    f'</div></div>'
                    f'<div style="margin-top: 8px; color: #6b7280; font-size: 0.85rem;">'
                    f'Avg confidence: <strong>{avg_c:.0%}</strong> · '
                    f'Max confidence: <strong>{max_c:.0%}</strong>'
                    f'</div>'
                    f'<div style="margin-top: 6px; color: #78716c; font-size: 0.85rem;">'
                    f'{info["description"]}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        elif total_detections == 0:
            st.warning("⚠️ No cotton growth stages detected. Try lowering the confidence threshold in the sidebar, or upload a different image.")

    else:
        # Empty state
        st.markdown("")
        col_empty = st.columns([1, 2, 1])[1]
        with col_empty:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: linear-gradient(145deg, #f0fdf4, #ecfdf5);
                        border-radius: 20px; border: 2px dashed #86efac;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🌿📸</div>
                <h3 style="color: #065f46; margin: 0;">Upload a Cotton Plant Image</h3>
                <p style="color: #6b7280; margin-top: 0.5rem;">
                    Drag & drop or click above to upload a photo.<br>
                    The AI will identify cotton growth stages instantly.
                </p>
                <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                    <span style="background: white; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; color: #6b7280; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        🌸 Cotton Blossom
                    </span>
                    <span style="background: white; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; color: #6b7280; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        🌱 Cotton Bud
                    </span>
                    <span style="background: white; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; color: #6b7280; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        🟢 Early Boll
                    </span>
                    <span style="background: white; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; color: #6b7280; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        🟤 Matured Boll
                    </span>
                    <span style="background: white; padding: 6px 14px; border-radius: 20px; font-size: 0.8rem; color: #6b7280; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        ☁️ Split Boll
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("")
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: #9ca3af; font-size: 0.8rem;">'
        '🌿 Cotton Boll Growth Detection · Powered by YOLOv8 · Built with Streamlit'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
