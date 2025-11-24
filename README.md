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

---

## 📂 Project Structure

SENTINEL-AI/
├── app.py # Main Streamlit Dashboard
├── generate_crowd_data.py # Synthetic dataset creation
├── prepare_data.py # Preprocessing → sequences (.npy)
├── train_model.py # LSTM model training pipeline
├── crowd_data.csv # Base dataset
├── requirements.txt # Dependencies
│
├── Models/
│ ├── lstm_crowd_behavior.h5 # Trained LSTM model
│ ├── yolov8n.pt # YOLOv8 weights
│ └── yolov8n-pose.pt # YOLO-Pose weights
│
└── Data_Artifacts/ # Auto-generated
├── X_train.npy
├── y_train.npy
└── label_encoder_classes.npy

---

## ⚡ Installation & Setup

### 1️⃣ Clone the Repository

git clone https://github.com/Ayushpk01/SENTINEL-AI.git
cd SENTINEL-AI
2️⃣ Install Dependencies
It is recommended to use a virtual environment:

bash
Copy code
pip install -r requirements.txt
3️⃣ Run the System
Launch the interactive dashboard:

bash
Copy code
streamlit run app.py
🧠 Model Workflow (Pipeline)
1. Feature Extraction
📌 For each frame:

Density = count of people / frame area

Velocity = average movement (optical flow + tracking ID history)

Pose Variance = deviation of key joint angles (shoulders, legs, neck)

2. Sequence Aggregation
Frames are stored in a sliding window

Sequence length = 10 frames

Shape → (1, 10, 3) → [density, motion, pose]

3. LSTM Prediction
Model predicts crowd state based on past 10 frames

Output → Calm | Dispersing | Aggressive | Stampede

4. RL Adaptive Tuning
Q-Learning agent monitors:

Stability

False positives/negatives

Adjusts thresholds dynamically

🔮 Future Roadmap
 RTSP (IP Camera) integration

 Email/SMS alerting via Twilio

 3D crowd density mapping

 Jetson Nano edge deployment

 Turbulence & panic-wave detection

👤 Author
Ayush PK
GitHub: https://github.com/Ayushpk01

