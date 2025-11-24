# 🛡️ SENTINEL AI – Autonomous Crowd Safety System

**Sentinel AI** is an advanced real-time **crowd analysis** and **anomaly detection** system.  
It leverages **Computer Vision**, **Deep Learning**, and **Reinforcement Learning** to detect early signs of **stampedes, chaos, aggression, and crowd instability**.

---

## 🚀 Key Features

### 🔍 Real-Time Detection
- **YOLOv8** → Person detection & tracking  
- **YOLO-Pose (Keypoints)** → Body movement & limb-angle variance  
- **Optical Flow + ByteTrack** → Motion velocity & crowd flow direction  

### 🧠 Behavior Classification (LSTM-Based)
The LSTM model classifies crowd behavior into **4 states**:

| State | Meaning |
|-------|---------|
| 🟢 **Calm** | Normal, smooth crowd flow |
| 🔵 **Dispersing** | People moving out of an area |
| 🟡 **Aggressive** | Chaotic behavior, fights, erratic motion |
| 🔴 **Stampede** | High speed + high density + unidirectional movement |

### 🤖 Adaptive Sensitivity (RL Agent)
A **Q-Learning agent** automatically adjusts:
- Density thresholds  
- Motion sensitivity  
- Pose variance tolerance  

This reduces **false positives** and improves reliability.

### 🖥️ Live Dashboard
A modern **Streamlit “Glassmorphism” UI** with:
- Real-time video feed  
- Heatmaps  
- Crowd density graphs  
- Alerts & telemetry panel  

---

## 🛠️ Tech Stack

### **Language**
- Python 3.x

### **Computer Vision**
- Ultralytics YOLOv8  
- YOLO-Pose  
- OpenCV  
- Supervision Toolkit  

### **Deep Learning**
- TensorFlow / Keras  
- LSTM Neural Network  

### **Reinforcement Learning**
- Q-Learning Algorithm

### **Frontend**
- Streamlit  

### **Utilities**
- NumPy  
- Pandas  
- Scikit-Learn  

## ⚡ How to Run Sentinel AI

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Ayushpk01/SENTINEL-AI.git
cd SENTINEL-AI
pip install -r requirements.txt
streamlit run app.py
python prepare_lstm_data.py
python train_lstm.py
streamlit run analyze_crowd.py


