🛡️ SENTINEL AI - Autonomous Crowd Safety System

Sentinel AI is an advanced crowd analysis and anomaly detection system. It leverages Computer Vision (YOLOv8 & Pose Estimation), Deep Learning (LSTM), and Reinforcement Learning (Q-Learning) to monitor video feeds in real-time, detecting potential threats like stampedes or aggressive behavior before they escalate.

🚀 Key Features

Real-Time Detection: Uses YOLOv8 to track individuals and YOLO-Pose to analyze body language variance.

Behavior Classification: Classifies crowd state into 4 distinct categories using an LSTM Neural Network:

🟢 Calm: Normal flow.

🔵 Dispersing: Crowd leaving an area.

🟡 Aggressive: Erratic movements, fighting, or high pose variance.

🔴 Stampede: High speed, high density, unidirectional flow.

Adaptive Sensitivity: Features a Reinforcement Learning (RL) Agent that automatically adjusts detection thresholds based on environmental stability.

Live Dashboard: A "Glassmorphism" UI built with Streamlit featuring real-time telemetry graphs, heatmaps, and alert systems.

🛠️ Tech Stack

Language: Python 3.x

Vision: Ultralytics YOLOv8, OpenCV, Supervision

Deep Learning: TensorFlow/Keras (LSTM)

Data Processing: NumPy, Pandas, Scikit-Learn

Interface: Streamlit

📂 Project Structure

SENTINEL-AI/
├── app.py                     # Main Sentinel AI Dashboard (Streamlit)
├── generate_crowd_data.py     # Generates synthetic training data
├── prepare_data.py            # Preprocesses CSV data into sequences (.npy)
├── train_model.py             # Trains the LSTM model
├── crowd_data.csv             # The dataset used for training
├── requirements.txt           # Dependencies
│
├── Models/
│   ├── lstm_crowd_behavior.h5    # Trained LSTM Model
│   ├── yolov8n.pt                # Object Detection Weights
│   └── yolov8n-pose.pt           # Pose Estimation Weights
│
└── Data_Artifacts/            # Generated during preprocessing
    ├── X_train.npy
    ├── y_train.npy
    └── label_encoder_classes.npy


⚡ Installation & Setup

Clone the Repository

git clone [https://github.com/Ayushpk01/SENTINEL-AI.git](https://github.com/Ayushpk01/SENTINEL-AI.git)
cd SENTINEL-AI



Install Dependencies
It is recommended to use a virtual environment.

pip install -r requirements.txt


Run the System
To launch the dashboard:

streamlit run app.py


🧠 Model Workflow

The system operates in a 4-stage pipeline:

Feature Extraction:

Density: Calculated via person count vs. frame area.

Flow Velocity: Average pixel distance moved per ID (ByteTrack).

Pose Variance: Analyzing shoulder/limb angles for chaotic movement.

Sequence Aggregation:

Data is stored in a sliding window buffer (Sequence Length = 10 frames).

Prediction (LSTM):

The sequence (1, 10, 3) is fed into the LSTM model to predict the next state.

Adaptive Tuning (RL):

A Q-Learning agent monitors the stability of the system. If false positives occur, it adjusts the sensitivity thresholds dynamically.

📸 Screenshots

        



🔮 Future Roadmap

[ ] Integration with live IP Cameras (RTSP).

[ ] Email/SMS Alert system via Twilio.

[ ] 3D Crowd Density mapping.

[ ] Deployment on Edge Devices (Jetson Nano).



Author: Ayushpk01
