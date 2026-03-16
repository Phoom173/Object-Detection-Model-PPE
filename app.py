import streamlit as st
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av # สำหรับจัดการ frame วิดีโอ

# --- 1. SETTINGS & THEME ---
st.set_page_config(
    page_title="SafetyAI | Construction Monitor",
    page_icon="👷",
    layout="wide"
)

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
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL (Cached) ---
@st.cache_resource
def load_yolo_model():
    try:
        # มั่นใจว่าไฟล์ best.pt อยู่ในโฟลเดอร์เดียวกับ app.py บน GitHub
        model = YOLO('best.pt') 
        return model
    except Exception as e:
        st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
        return None

model = load_yolo_model()

# --- 3. RTC CONFIGURATION (สำหรับข้าม Firewall บน Cloud) ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- 4. VIDEO PROCESSING CLASS ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.conf_threshold = 0.5

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if model is not None:
            # รันการตรวจจับ
            results = model.predict(img, conf=self.conf_threshold, verbose=False)
            # วาดกรอบ Bounding Box ลงบนภาพ
            annotated_img = results[0].plot()
            
            # (Optional) ตรงนี้คือจุดที่คุณสามารถใส่ Logic แจ้งเตือนได้ในอนาคต
        else:
            annotated_img = img

        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🚧 Safety Control")
    st.write("---")
    app_mode = st.radio("เลือกโหมดการทำงาน", ["Dashboard", "Live Monitoring"])
    st.write("---")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.info("Status: System Ready")

# --- 6. MAIN PAGE LOGIC ---
if app_mode == "Dashboard":
    st.title("👷 Construction Safety Dashboard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Workers", "Checking...", "Live")
    m2.metric("Compliance", "90%", "Target: 100%")
    m3.metric("Alerts Today", "0", "Good")
    m4.metric("System", "Online")

elif app_mode == "Live Monitoring":
    st.title("📽️ Real-time Detection (WebRTC)")
    
    col_vid, col_stat = st.columns([7, 3])
    
    with col_vid:
        st.markdown("#### Camera Feed")
        # ส่วนเรียกใช้งานกล้องผ่าน WebRTC
        ctx = webrtc_streamer(
            key="ppe-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # ส่งค่า Confidence จาก Slider เข้าไปใน Processor
        if ctx.video_processor:
            ctx.video_processor.conf_threshold = conf_threshold

    with col_stat:
        st.markdown("#### Analysis Details")
        if ctx.state.playing:
            st.success("กล้องกำลังทำงาน...")
        else:
            st.warning("กด START ด้านซ้ายเพื่อเริ่มกล้อง")