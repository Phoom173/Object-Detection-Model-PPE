import streamlit as st
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av

# --- 1. การตั้งค่าหน้าเว็บ (UI/UX Design) ---
st.set_page_config(
    page_title="SafetyAI | Construction Monitor",
    page_icon="👷",
    layout="wide"
)

# Custom CSS สำหรับโหมด Premium (High-Contrast)
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { 
        background-color: #1f2937; 
        padding: 15px; 
        border-radius: 12px; 
        border-bottom: 4px solid #FFD700;
    }
    div[data-testid="stExpander"] { border: none; background: #1f2937; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. การโหลดโมเดล (Cached) ---
@st.cache_resource
def load_yolo_model():
    try:
        # ใช้ไฟล์ best.pt ที่คุณเทรนเสร็จวางไว้ในโฟลเดอร์เดียวกัน
        model = YOLO('best.pt') 
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_yolo_model()

# --- 3. การตั้งค่าระบบวิดีโอ (WebRTC & STUN) ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- 4. หัวใจหลัก: ส่วนประมวลผลวิดีโอ (AI Logic) ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.conf_threshold = 0.5
        # เก็บจำนวนที่ตรวจเจอ (สำหรับแสดงผลที่ Sidebar/Dashboard)
        self.results_data = {"Worker": 0, "Helmet": 0, "Vest": 0, "Gloves": 0, "Boots": 0}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if model is not None:
            # รัน YOLO (ลดขนาด imgsz เพื่อเพิ่ม FPS บน Cloud)
            results = model.predict(img, conf=self.conf_threshold, verbose=False, imgsz=480)
            annotated_img = results[0].plot()
            
            # นับจำนวน Class ที่เจอในเฟรมปัจจุบัน
            temp_counts = {"Worker": 0, "Helmet": 0, "Vest": 0, "Gloves": 0, "Boots": 0}
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                if label in temp_counts:
                    temp_counts[label] += 1
            
            self.results_data = temp_counts
        else:
            annotated_img = img

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- 5. การจัดวางหน้าจอ (Layout) ---
st.title("👷 AI-Powered Construction Safety Monitoring")

# Sidebar สำหรับควบคุม
with st.sidebar:
    st.header("⚙️ Control Panel")
    app_mode = st.radio("เลือกหน้าจอ", ["Live Monitoring", "System Dashboard"])
    st.write("---")
    conf_threshold = st.slider("ความแม่นยำ (Confidence)", 0.0, 1.0, 0.5)
    facing_mode = st.selectbox("เลือกกล้อง (สำหรับมือถือ)", ["environment", "user"], index=0)
    st.info("Status: System Running")

if app_mode == "Live Monitoring":
    col_vid, col_stat = st.columns([7, 3])
    
    with col_vid:
        st.subheader("📽️ Live Feed")
        ctx = webrtc_streamer(
            key="ppe-check",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {"facingMode": facing_mode},
                "audio": False
            },
            async_processing=True,
        )

    with col_stat:
        st.subheader("📊 Real-time Stats")
        if ctx.video_processor:
            # ดึงข้อมูลจาก Processor มาแสดงผลข้างวิดีโอ
            data = ctx.video_processor.results_data
            ctx.video_processor.conf_threshold = conf_threshold
            
            st.metric("Workers Detected", data["Worker"])
            st.metric("Safety Helmets", data["Helmet"])
            st.metric("Safety Vests", data["Vest"])
            
            # Logic แจ้งเตือนพื้นฐาน
            if data["Worker"] > data["Helmet"]:
                st.error(f"🚨 ALERT: พบคนงาน {data['Worker'] - data['Helmet']} คน ไม่ใส่หมวก!")
            if data["Worker"] > data["Vest"]:
                st.warning(f"⚠️ Warning: พบคนงาน {data['Worker'] - data['Vest']} คน ไม่ใส่เสื้อกั๊ก")
        else:
            st.write("กรุณากด START เพื่อเริ่มระบบตรวจจับ")

elif app_mode == "System Dashboard":
    st.subheader("📈 Safety Analytics Summary")
    # ส่วนนี้สามารถดึงฐานข้อมูลมาแสดงเป็นกราฟในอนาคต
    st.info("ส่วนแสดงสถิติภาพรวมความปลอดภัยรายวัน")
    c1, c2, c3 = st.columns(3)
    c1.metric("Compliance Rate", "85%", "+5%")
    c2.metric("Total Incidents", "2", "-1")
    c3.metric("Inspected Workers", "124", "Today")