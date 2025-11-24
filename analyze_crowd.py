import streamlit as st
from ultralytics import YOLO
import cv2
import supervision as sv
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from tensorflow.keras.models import load_model
import tempfile
import os
import random
import math
import time

# --- 1. Page Configuration & Ultra-Modern UI ---
st.set_page_config(
    layout="wide", 
    page_title="SENTINEL AI👁️", 
    page_icon="👁️",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Glassmorphism" and Professional Dashboard
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(17, 24, 39) 0%, rgb(0, 0, 0) 90%);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b0f19;
        border-right: 1px solid #1f2937;
    }

    /* Custom Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(31, 41, 55, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(75, 85, 99, 0.4);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    div[data-testid="stMetricLabel"] {
        color: #9ca3af;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #f3f4f6;
        font-size: 2rem;
        font-weight: 700;
        font-family: 'SF Mono', 'Roboto Mono', monospace;
    }

    /* Charts Background */
    [data-testid="stChart"] {
        background: rgba(31, 41, 55, 0.3);
        border-radius: 12px;
        padding: 10px;
        border: 1px solid rgba(75, 85, 99, 0.2);
    }

    /* Headers */
    h1, h2, h3 {
        color: #f3f4f6 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(to right, #2563eb, #3b82f6);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #1d4ed8, #2563eb);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }

    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Load Models & Resources ---
@st.cache_resource
def load_models():
    # Load YOLO models (Standard + Pose)
    yolo_model = YOLO("yolov8n.pt")
    pose_model = YOLO("yolov8n-pose.pt")
    
    lstm_model = None
    label_classes = ["Calm", "Dispersing", "Aggressive"] # Default fallback
    
    # Load Custom LSTM (Matches your training code)
    try:
        if os.path.exists("lstm_crowd_behavior.h5"):
            lstm_model = load_model("lstm_crowd_behavior.h5")
            print("LSTM Model Loaded Successfully")
        
        if os.path.exists("label_encoder_classes.npy"):
            label_classes = np.load("label_encoder_classes.npy", allow_pickle=True)
    except Exception as e:
        print(f"Custom model not found, using rule-based fallback. Error: {e}")
        
    return yolo_model, pose_model, lstm_model, label_classes

yolo_model, pose_model, lstm_model, label_classes = load_models()

# --- 3. Logic Classes ---

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.q_table = {}
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate

    def get_state(self, density, speed, pose_variance):
        # Discretize continuous values into bins for the Q-Table
        d_bin = int(min(density * 100, 10))  # 0-10
        s_bin = int(min(speed, 20) // 2)     # 0-10
        p_bin = int(min(pose_variance, 50) // 5) # 0-10
        return (d_bin, s_bin, p_bin)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        # Epsilon-Greedy Strategy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        q_values = [self.get_q_value(state, a) for a in self.actions]
        # If all Q-values are 0 (unexplored), pick random, else pick max
        if all(q == 0 for q in q_values):
            return random.choice(self.actions)
        return self.actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        # Bellman Equation
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
        new_q = old_q + self.lr * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(state, action)] = new_q

# --- 4. Processing Logic ---

def generate_heatmap(frame, centroids, intensity=50):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    for x, y in centroids:
        cv2.circle(heatmap, (x, y), 40, intensity, -1)
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Global Tracker
tracker = sv.ByteTrack()

def process_video_feed(source, rl_agent, thresholds, sequence_buffer, history_buffers):
    cap = cv2.VideoCapture(source)
    prev_positions = defaultdict(lambda: None)
    
    # --- UI Layout for Video & Analytics ---
    # Top Row: KPIs
    kpi_container = st.container()
    
    # Middle: Video and Charts split
    col_main, col_sidebar = st.columns([2, 1])
    
    with col_main:
        video_slot = st.empty()
    
    with col_sidebar:
        st.markdown("##### 📈 Live Telemetry")
        chart_density = st.empty()
        st.markdown("##### ⚡ Flow Velocity")
        chart_speed = st.empty()
        
    # Alert Area (Bottom)
    alert_slot = st.empty()
    
    # Live RL Monitor in Sidebar
    rl_monitor = st.sidebar.empty()
    
    # Initialize UI elements inside container
    with kpi_container:
        k1, k2, k3, k4 = st.columns(4)
        kpi_count = k1.empty()
        kpi_density = k2.empty()
        kpi_speed = k3.empty()
        kpi_status = k4.empty()
    
    # State tracking
    status = "Calm"
    frame_count = 0
    
    while cap.isOpened() and st.session_state.run_analysis:
        ret, frame = cap.read()
        if not ret: break
        
        # Resize for performance (720p is good balance)
        frame = cv2.resize(frame, (854, 480))
        annotated_frame = frame.copy()
        
        # ---------------------------
        # 1. FEATURE EXTRACTION
        # ---------------------------
        
        # A. YOLO Person Detection
        results = yolo_model(frame, classes=[0], verbose=False, conf=0.3)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)
        
        # B. Pose Detection (for Variance)
        pose_results = pose_model(frame, verbose=False)
        keypoints = pose_results[0].keypoints.xy.cpu().numpy() if pose_results[0].keypoints is not None else []
        
        # C. Calculate Metrics
        num_people = len(detections)
        centroids = []
        speeds = []
        pose_angles = []
        
        # Velocity Calculation
        for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            centroids.append((cx, cy))
            
            # Speed from previous frame
            if prev_positions[track_id] is not None:
                px, py = prev_positions[track_id]
                dist = math.sqrt((cx-px)**2 + (cy-py)**2)
                speeds.append(dist)
                
                # Draw Flow Vectors
                if dist > 2:
                    cv2.arrowedLine(annotated_frame, (px, py), (cx, cy), (0, 255, 255), 2)
            
            prev_positions[track_id] = (cx, cy)
            # Draw Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 100), 2)

        # Pose Variance Calculation (Shoulder tilt)
        for kp in keypoints:
            if len(kp) >= 7: # Ensure shoulders exist
                ls, rs = kp[5], kp[6] # Left/Right Shoulder
                if ls[0] > 0 and rs[0] > 0:
                    angle = math.atan2(rs[1]-ls[1], rs[0]-ls[0])
                    pose_angles.append(angle)

        # ---------------------------
        # 2. DATA AGGREGATION
        # ---------------------------
        
        # Current Frame Features
        # FIX: Density scaling adapted for visual feedback
        current_density = (num_people / (frame.shape[0]*frame.shape[1])) * 10000 
        current_speed = np.mean(speeds) if speeds else 0
        current_pose_var = np.var(pose_angles) * 100 if pose_angles else 0

        # Append to Sliding Window (The Sequence for LSTM)
        sequence_buffer.append([current_density, current_speed, current_pose_var])
        
        # Update History for Charts (deque handles memory automatically)
        history_buffers['density'].append(current_density)
        history_buffers['speed'].append(current_speed)
        
        # ---------------------------
        # 3. INTELLIGENCE (LSTM + RL)
        # ---------------------------
        
        # A. LSTM Prediction
        if lstm_model and len(sequence_buffer) == 10:
            # Reshape to (1, 10, 3) as per your training script
            input_seq = np.array(sequence_buffer).reshape(1, 10, 3)
            prediction = lstm_model.predict(input_seq, verbose=0)
            status = label_classes[np.argmax(prediction)]
        else:
            # Fallback Rule-Based Logic (FIXED SENSITIVITY)
            # Thresholds are dynamically tuned by RL, but defaults are now sane.
            
            # Logic: If speed exceeds dynamic threshold -> Stampede
            if current_speed > thresholds['speed'] * 1.5: 
                status = "Stampede"
            # Logic: If speed is moderate BUT Pose variance is high -> Aggressive
            elif current_speed > thresholds['speed'] * 0.8 and current_pose_var > 10.0: 
                status = "Aggressive"
            # Logic: Very low density -> Dispersing
            elif current_density < 0.2 and num_people > 0: 
                status = "Dispersing"
            else: 
                status = "Calm"

        # B. Reinforcement Learning Update
        # 1. Get current state
        rl_state = rl_agent.get_state(current_density, current_speed, current_pose_var)
        
        # 2. Choose Action (Modify Thresholds)
        action = rl_agent.choose_action(rl_state)
        
        # 3. Apply Action
        if action == "increase_sensitivity": 
            thresholds['speed'] = max(1.5, thresholds['speed'] - 0.1) # Lower threshold = Higher sensitivity
        elif action == "decrease_sensitivity": 
            thresholds['speed'] = min(8.0, thresholds['speed'] + 0.1) # Cap max at 8.0, not 20.0
        
        # 4. Calculate Reward (FIXED: Don't punish for detection)
        # Goal: Maintain threshold near an optimal '3.0' unless environment dictates otherwise.
        # This prevents the agent from exploding the threshold to 20.0 just to be "Calm".
        target_optimal = 3.0 
        reward = 1.0 - (abs(thresholds['speed'] - target_optimal) / 10.0)
        
        # 5. Get Next State & Learn
        next_rl_state = rl_agent.get_state(current_density, current_speed, current_pose_var)
        rl_agent.update_q_table(rl_state, action, reward, next_rl_state)

        # Update RL Sidebar Monitor
        rl_monitor.info(f"**Adaptive Threshold:** {thresholds['speed']:.2f} px/f")

        # ---------------------------
        # 4. VISUALIZATION
        # ---------------------------
        
        # Overlay Heatmap
        heatmap = generate_heatmap(frame, centroids)
        final_view = cv2.addWeighted(annotated_frame, 0.7, heatmap, 0.4, 0)
        
        # Display Video
        video_slot.image(final_view, channels="BGR", use_column_width=True)
        
        # Update KPIs with Custom Cards styling (via native metric, enhanced by CSS)
        kpi_count.metric("Active Agents", num_people)
        kpi_density.metric("Density Index", f"{current_density:.2f}")
        kpi_speed.metric("Flow Velocity", f"{current_speed:.1f}")
        
        # Custom Status Logic
        if status == "Stampede":
            kpi_status.markdown(
                """
                <div style='background-color: rgba(239, 68, 68, 0.2); border: 1px solid #ef4444; border-radius: 12px; padding: 15px; text-align: center;'>
                    <div style='color: #ef4444; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; margin-bottom: 5px;'>System Status</div>
                    <div style='color: #f87171; font-size: 1.5rem; font-weight: 800;'>CRITICAL</div>
                </div>
                """, unsafe_allow_html=True
            )
            alert_slot.error("🚨 STAMPEDE PROTOCOLS INITIATED: Mass panic vectors detected.")
        elif status == "Aggressive":
             kpi_status.markdown(
                """
                <div style='background-color: rgba(234, 179, 8, 0.2); border: 1px solid #eab308; border-radius: 12px; padding: 15px; text-align: center;'>
                    <div style='color: #eab308; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; margin-bottom: 5px;'>System Status</div>
                    <div style='color: #facc15; font-size: 1.5rem; font-weight: 800;'>WARNING</div>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            kpi_status.markdown(
                f"""
                <div style='background-color: rgba(34, 197, 94, 0.2); border: 1px solid #22c55e; border-radius: 12px; padding: 15px; text-align: center;'>
                    <div style='color: #22c55e; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; margin-bottom: 5px;'>System Status</div>
                    <div style='color: #4ade80; font-size: 1.5rem; font-weight: 800;'>{status.upper()}</div>
                </div>
                """, unsafe_allow_html=True
            )
            alert_slot.empty()

        # Update Charts
        frame_count += 1
        if frame_count % 5 == 0:
            chart_density.line_chart(list(history_buffers['density']), height=150)
            chart_speed.line_chart(list(history_buffers['speed']), height=150)

    cap.release()

# --- 5. Main Layout ---

def main():
    # Sidebar
    st.sidebar.title("⚙️ Control Center")
    st.sidebar.markdown("---")
    
    # Input Selection
    st.sidebar.subheader("Video Source")
    input_type = st.sidebar.selectbox("Select Feed", ["Webcam", "Video Upload"], index=0)
    video_source = 0
    if input_type == "Video Upload":
        uploaded_file = st.sidebar.file_uploader("Upload MP4", type=["mp4", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_source = tfile.name

    st.sidebar.markdown("---")

    st.sidebar.subheader("PBLTEAM:CODE OF DUTY")

    st.sidebar.info("BUILT by AYUSH,DHANUSH,NEEMA AND NIREEKSHA")
    
    

    # Initialization
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False

    st.sidebar.markdown("---")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_btn = st.button("▶ START", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹ STOP", use_container_width=True)

    if start_btn:
        st.session_state.run_analysis = True
    if stop_btn:
        st.session_state.run_analysis = False

    # Header Section
    st.title(" SENTINEL AI👁️")
    st.markdown("""
    <div style='margin-bottom: 20px; color: #9ca3af; font-size: 1.1rem;'>
    Autonomous Crowd Safety & Anomaly Detection System
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.run_analysis:
        # Initialize RL Agent
        actions = ["increase_sensitivity", "decrease_sensitivity", "maintain"]
        agent = QLearningAgent(actions)
        
        # Initialize Buffers
        # FIX: Lower default threshold from 5.0 to 3.0 to catch movement easier
        thresholds = {'speed': 3.0} 
        sequence_buffer = deque(maxlen=10) # For LSTM (Last 10 frames)
        history_buffers = {
            'density': deque(maxlen=100), # For Charts (Last 100 frames)
            'speed': deque(maxlen=100)
        }
        
        process_video_feed(video_source, agent, thresholds, sequence_buffer, history_buffers)

if __name__ == "__main__":
    main()