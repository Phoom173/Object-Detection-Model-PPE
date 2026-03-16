import streamlit as st
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av

# --- 1. SETTINGS & THEME (ปรับปรุงสีให้ชัดเจน) ---
st.set_page_config(
    page_title="SafetyAI | Construction Monitor",
    page_icon="👷",
    layout="wide"
)

st.markdown("""
    <style>
    /* พื้นหลังหลัก */
    .main { background-color: #0b0e14; color: white; }
    
    /* Sidebar: ปรับให้สีเข้มกว่าพื้นหลังหลักเพื่อความชัดเจน */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Metric Card: ปรับตัวอักษรเป็นสีขาวสว่าง */
    [data-testid="stMetric"] { 
        background-color: #1c2128; 
        padding: 15px; 
        border-radius: 12px; 
        border-bottom: 4px solid #FFD700;
    }
    
    /* บังคับสีตัวอักษร Metric ทั้งหมดให้เป็นสีขาว */
    [data-testid="stMetricLabel"] p {
        color: #ffffff !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricValue"] div {
        color: #ffffff !important;
        font-size: 28px !important;
    }
    
    /* ปรับแต่งหัวข้อ */
    h1, h2, h3 { color: #FFD700 !important; }
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

# --- 3. VIDEO PROCESSING (เพิ่มระบบ Frame Skipping เพื่อความลื่นไหล) ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.conf_threshold = 0.5
        self.frame_count = 0
        self.skip_frames = 2  # ประมวลผล 1 เฟรม และข้าม 2 เฟรม (ปรับเพิ่มได้ถ้ายังกระตุก)
        self.last_results = {"Person": 0, "Helmet": 0, "Vest": 0, "Boots": 0, "Gloves": 0}
        self.last_annotated = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # รัน AI เฉพาะเฟรมที่ไม่ได้ถูกข้าม
        if self.frame_count % (self.skip_frames + 1) == 0 or self.last_annotated is None:
            if model is not None:
                results = model.predict(img, conf=self.conf_threshold, verbose=False, imgsz=480)
                self.last_annotated = results[0].plot()
                
                # อัปเดตตัวนับ
                temp_counts = {"Person": 0, "Helmet": 0, "Vest": 0, "Boots": 0, "Gloves": 0}
                for box in results[0].boxes:
                    label = model.names[int(box.cls[0])]
                    if label in temp_counts:
                        temp_counts[label] += 1
                self.last_results = temp_counts
            else:
                self.last_annotated = img

        return av.VideoFrame.from_ndarray(self.last_annotated, format="bgr24")

# --- 4. LAYOUT & SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    app_mode = st.radio("เลือกหน้าจอ", ["Live Monitoring", "System Dashboard"])
    st.write("---")
    conf_threshold = st.slider("ความแม่นยำ (Confidence)", 0.0, 1.0, 0.5)
    facing_mode = st.selectbox("เลือกกล้อง", ["environment", "user"], index=0)
    st.success("System: Ready")

# --- 5. MAIN LOGIC ---
if app_mode == "Live Monitoring":
    st.title("📽️ Safety Monitoring (Optimized)")
    
    col_vid, col_stat = st.columns([7, 3])
    
    with col_vid:
        ctx = webrtc_streamer(
            key="ppe-check",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": {"facingMode": facing_mode}, "audio": False},
            async_processing=True,
        )

    with col_stat:
        if ctx.video_processor:
            data = ctx.video_processor.last_results
            ctx.video_processor.conf_threshold = conf_threshold
            
            st.metric("Persons Detected", data["Person"])
            st.metric("Safety Helmets", data["Helmet"])
            st.metric("Safety Vests", data["Vest"])
            st.metric("Boots/Gloves", f"{data['Boots']} / {data['Gloves']}")
            
            # Logic แจ้งเตือน
            if data["Person"] > data["Helmet"]:
                st.error(f"🚨 ALERT: {data['Person'] - data['Helmet']} คน ไม่ใส่หมวก")
            if data["Person"] > data["Vest"]:
                st.warning(f"⚠️ WARNING: {data['Person'] - data['Vest']} คน ไม่ใส่เสื้อกั๊ก")
        else:
            st.info("กรุณากด START เพื่อเริ่มกล้อง")

elif app_mode == "System Dashboard":
    st.title("📈 Analytics Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Compliance Rate", "92%", "+2%")
    c2.metric("Total Persons", "45", "Live")
    c3.metric("Critical Alerts", "0", "Good")