import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# --- 1. SETTINGS & THEME ---
st.set_page_config(
    page_title="SafetyAI | Construction Monitor",
    page_icon="👷",
    layout="wide"
)

# ปรับปรุง CSS เพื่อให้ตัวอักษรใน Metric เป็นสีขาวทั้งหมด
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    
    /* แก้ไขพื้นหลังของ Metric Card */
    [data-testid="stMetric"] { 
        background-color: #1f2937; 
        padding: 15px; 
        border-radius: 12px; 
        border-bottom: 4px solid #FFD700;
        color: white; /* บังคับสีขาว */
    }

    /* บังคับสีตัวอักษรของ Label (หัวข้อ metric) */
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-weight: bold;
    }

    /* บังคับสีตัวอักษรของ Value (ตัวเลข metric) */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    .stSidebar { background-color: #111827; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { 
        background-color: #1f2937; 
        padding: 15px; 
        border-radius: 12px; 
        border-bottom: 4px solid #FFD700;
    }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7d32,#005005); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO('best.pt') 
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_yolo_model()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    app_mode = st.radio("เลือกแหล่งข้อมูล", ["Webcam (Live)", "Upload Video", "Dashboard Summary"])
    st.write("---")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.info("Classes: Boots, Gloves, Helmet, Person, Vest")

# --- 4. CORE PROCESSING FUNCTION ---
def run_detection(video_source):
    cap = cv2.VideoCapture(video_source)
    
    col_vid, col_stat = st.columns([7, 3])
    with col_vid:
        frame_window = st.image([])
    with col_stat:
        st.subheader("📊 Live Statistics")
        stats_container = st.empty()
        alert_container = st.empty()

    stop_btn = st.button("Stop Processing")

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model.predict(frame, conf=conf_threshold, verbose=False, imgsz=300)
        annotated_frame = results[0].plot()

        # Count PPE Classes
        current_counts = {"Person": 0, "Helmet": 0, "Vest": 0, "Boots": 0, "Gloves": 0}
        
        for box in results[0].boxes:
            label = model.names[int(box.cls[0])]
            if label in current_counts:
                current_counts[label] += 1

        # Display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_window.image(annotated_frame, use_container_width=True)

        # Update Metrics
        with stats_container.container():
            s1, s2 = st.columns(2)
            s1.metric("Persons", current_counts["Person"])
            s2.metric("Helmets", current_counts["Helmet"])
            s3, s4 = st.columns(2)
            s3.metric("Vests", current_counts["Vest"])
            s4.metric("Boots", current_counts["Boots"])
            st.metric("Gloves (Pairs/Items)", current_counts["Gloves"])

        # Alert Logic (ตรวจสอบอุปกรณ์พื้นฐาน)
        with alert_container.container():
            if current_counts["Person"] > 0:
                missing_helmet = max(0, current_counts["Person"] - current_counts["Helmet"])
                missing_vest = max(0, current_counts["Person"] - current_counts["Vest"])
                
                if missing_helmet > 0:
                    st.error(f"🚨 ALERT: Missing Helmet ({missing_helmet} Person)")
                if missing_vest > 0:
                    st.warning(f"⚠️ Warning: Missing Vest ({missing_vest} Person)")
                if missing_helmet == 0 and missing_vest == 0:
                    st.success("✅ PPE Compliance: High")

    cap.release()

# --- 5. MAIN LOGIC ---
if app_mode == "Webcam (Live)":
    st.title("📽️ Live Webcam Monitoring")
    if st.button("Start Camera"):
        run_detection(0)

elif app_mode == "Upload Video":
    st.title("📂 Video File Analysis")
    uploaded_file = st.file_uploader("Upload MP4/AVI", type=['mp4', 'avi', 'mov'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        if st.button("Analyze Video"):
            run_detection(tfile.name)

elif app_mode == "Dashboard Summary":
    st.title("📈 Safety Performance Dashboard")
    st.write("สถิติภาพรวมจากการประมวลผล")
    # ตัวอย่างกราฟจำลองความปลอดภัย
    chart_data = np.random.randn(20, 3)
    st.line_chart(chart_data)
    st.success("ระบบพร้อมสำหรับการวิเคราะห์ข้อมูลย้อนหลัง")