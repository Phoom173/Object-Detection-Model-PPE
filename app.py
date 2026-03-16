import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- 1. SETTINGS & THEME ---
st.set_page_config(
    page_title="SafetyAI | Construction Monitor",
    page_icon="👷",
    layout="wide"
)

# Custom CSS เพื่อความ Premium และ High-Contrast
st.markdown("""
    <style>
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    .main { background-color: #0e1117; }
    .stMetric { 
        background-color: #1f2937; 
        padding: 20px; 
        border-radius: 12px; 
        border-bottom: 4px solid #FFD700;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL (Cached) ---
@st.cache_resource
def load_yolo_model():
    # เปลี่ยนเป็น path ของไฟล์ที่คุณเทรนเสร็จ เช่น 'runs/detect/train/weights/best.pt'
    try:
        model = YOLO('best.pt') 
        return model
    except:
        return None

model = load_yolo_model()

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🚧 Safety Control")
    st.write("---")
    app_mode = st.radio("เลือกโหมดการทำงาน", ["Dashboard", "Live Monitoring", "Reports"])
    st.write("---")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.success("System: Active")

# --- 4. MAIN PAGE LOGIC ---
if app_mode == "Dashboard":
    st.title("👷 Construction Safety Dashboard")
    
    # ส่วนสรุปตัวเลขด้านบน
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Workers", "15", "2 new")
    m2.metric("PPE Compliance", "94%", "High")
    m3.metric("Critical Alerts", "1", "-10%", delta_color="inverse")
    m4.metric("Active Cameras", "1", "Online")

    # ส่วนกราฟหรือข้อมูลจำลอง (คุณสามารถเพิ่ม Plotly ต่อได้ที่นี่)
    st.info("ยินดีต้อนรับสู่ระบบตรวจสอบความปลอดภัย โปรดเลือกโหมด Live Monitoring เพื่อเริ่มตรวจจับ")

elif app_mode == "Live Monitoring":
    st.title("📽️ Real-time Detection")
    
    col_vid, col_stat = st.columns([7, 3])
    
    with col_vid:
        st.markdown("#### Camera Feed")
        # สร้าง placeholder สำหรับวิดีโอ
        frame_window = st.image([], width="stretch")
        
        # ปุ่มเริ่ม/หยุด
        run_camera = st.checkbox('เปิดกล้อง (Start Camera)', value=False)
        
    with col_stat:
        st.markdown("#### Analysis Details")
        alert_placeholder = st.empty()
        st.write("---")
        log_placeholder = st.empty()

    # --- DETECTION LOOP ---
    if run_camera:
        # 0 คือกล้อง Webcam, หรือใส่ URL ของกล้อง IP ก็ได้
        cap = cv2.VideoCapture(0) 
        
        while cap.isOpened() and run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("ไม่สามารถเชื่อมต่อกล้องได้")
                break

            # 1. รัน YOLO Detection
            if model:
                results = model.predict(frame, conf=conf_threshold)
                annotated_frame = results[0].plot() # วาด Bounding Box
                
                # 2. แสดงผลใน Streamlit
                frame_window.image(annotated_frame, channels="BGR", width="stretch")
                
                # 3. Logic ตรวจสอบ PPE (ตัวอย่าง)
                # เราจะแกะผลลัพธ์จาก results[0].boxes เพื่อเช็คว่าใครไม่ใส่หมวกบ้าง
                # alert_placeholder.error("🚨 ALERT: Worker detected without Helmet!")
            else:
                st.warning("กำลังรอไฟล์โมเดล best.pt...")
                frame_window.image(frame, channels="BGR", width="stretch")

            # หน่วงเวลาเล็กน้อยเพื่อให้ระบบไม่ทำงานหนักเกินไป
            time.sleep(0.01)
        
        cap.release()

elif app_mode == "Reports":
    st.title("📋 Safety Incident Reports")
    st.write("ส่วนนี้สำหรับดาวน์โหลดข้อมูลย้อนหลัง")
    st.button("Download CSV Report")