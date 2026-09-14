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

## 💻 Local Setup & Running

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
   cd <REPO_NAME>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app**.
4. Select your repository, set the branch to `main`, and the main file path to `app.py`.
5. Click **Deploy!**
