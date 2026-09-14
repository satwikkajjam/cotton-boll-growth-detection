# 🌿 Cotton Boll Growth Detection using YOLOv8

An AI-powered computer vision application for detecting and classifying cotton boll growth stages using a custom-trained **YOLOv8** model. Built with **Streamlit** for interactive inference, visual analysis, and agricultural insights.

---

## 🎯 Supported Growth Stages (5 Classes)

| Stage | Name | Description |
| :---: | :--- | :--- |
| 🌸 | **Cotton Blossom** | White/cream-colored flowers indicating the active pollination phase. |
| 🌱 | **Cotton Bud** | Small green squares/buds forming before flowering. |
| 🟢 | **Early Boll** | Compact green bolls developing right after fertilization. |
| 🟤 | **Matured Cotton Boll** | Hardened bolls with fully developed fibers preparing to split. |
| ☁️ | **Split Cotton Boll** | Fluffy white cotton exposed and ready for harvesting. |

---

## 🚀 Model Performance
- **Base Architecture:** YOLOv8n (Detection)
- **Input Resolution:** 640 × 640
- **Peak Precision:** **82.1%**
- **Peak mAP50:** **49.4%**
- **Weights:** [`models/best_cotton_boll_model.pt`](models/best_cotton_boll_model.pt)

---
