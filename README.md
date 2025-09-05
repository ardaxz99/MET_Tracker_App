# Challenge 2025 — ADAMMA
**On-Device MET Class Prediction from Smartphone Accelerometers**

This repository contains the Android app, training code, exported models, demonstration video, and paper for our submission to **Challenge 2025 (ADAMMA)**. We estimate **MET classes** (Sedentary · Light · Moderate · Vigorous) from smartphone accelerometer data (WISDM dataset) and deploy a compact ONNX model on Android.

---

## 📂 Repository Layout

- `app/`  
  Kotlin Android app with a foreground service (`SensorManager`), feature computation, on-device inference, and live UI.

- `gradle/wrapper/`  
  Gradle wrapper scripts and configuration files for reproducible Android builds.

- `MET_Tracker_Trainer/`  
  Training & evaluation:
  - `main.py` (entry point)
  - `datasets/` (windowing, preprocessing)
  - `models/` (classic ML + 1D CNN)

- `video_onnx_and_apk_report/`  
  Packaged artifacts:
  - Android **APK**
  - **Video** demonstration of the app
  - ONNX-exported CNN model **`cnn.onnx`**
  - **PDF** report (this paper)

---

## ⚙️ Environment

- Python ≥ 3.9  
- PyTorch, scikit-learn, NumPy, pandas  
- (Optional) CUDA for faster CNN training  
- Android Studio (AGP 8.5.2, Gradle 8.7), `compileSdk=34`, `minSdk=24`

Install Python deps (from inside the trainer folder):
```bash
cd MET_Tracker_Trainer
pip install -r requirements.txt
```

## 🚀 Training

**Handcrafted features + classic ML**
```bash
cd MET_Tracker_Trainer
python main.py   --dataset WISDM   --dataset_path /path/to/WISDM/   --feature_extraction handcrafted   --model ml   --n_splits 5
```

**Raw windows + compact 1D CNN**
```bash
cd MET_Tracker_Trainer
python main.py   --dataset WISDM   --dataset_path /path/to/WISDM/   --feature_extraction raw   --model cnn   --n_splits 5   --batch_size 32
```

---

## 📱 Android App

- Continuous sensing at ~20 Hz; on-device inference (ONNX Runtime mobile).
- Foreground service keeps predictions running; UI shows current class.

**Install the APK:**
1. Enable *Install unknown apps* on your device.
2. Transfer and install:
   ```
   video_onnx_and_apk_report/<your-apk>.apk
   ```
3. Launch the app and press **Start** to begin sensing.

---

## 📦 Reproducibility & Artifacts

All artifacts are bundled for convenience:

- **APK** (ready to install)
- **Demo video** of the app running
- **`cnn.onnx`** exported model
- **Report (PDF)** — the paper corresponding to this repo

All four files live in:
```
video_onnx_and_apk_report/
```

---

## 📊 Results (summary)

- Models evaluated: Logistic Regression, kNN, RBF-SVM, Random Forest, Gaussian NB, MLP, and a compact **1D CNN**.
- The **1D CNN** on **raw windows** offers the best accuracy/latency trade-off for on-device deployment.
- Classic ML baselines work well on the **43-D handcrafted** features.

(See the PDF report for full tables and methodology.)
