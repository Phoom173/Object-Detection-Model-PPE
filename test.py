import streamlit as st

# 1. การตั้งค่าหน้ากระดาษให้ดูเป็น Web App ระดับมืออาชีพ
st.set_page_config(
    page_title="SafetyAI - Construction Monitor",
    page_icon="👷",
    layout="wide",
)

# 2. Custom CSS เพื่อความ Premium (สี Dark Theme + Safety Yellow)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border-left: 5px solid #FFD700; }
    h1 { color: #FFD700; font-family: 'Arial Black'; }
    </style>
    """, unsafe_allow_html=True)

# 3. Sidebar สำหรับตั้งค่า
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1087/1087121.png", width=100) # รูป Icon หมวก
    st.title("Control Panel")
    st.write("---")
    camera_option = st.selectbox("Select Camera Source", ["Integrated Webcam", "Mobile Camera (IP)", "Test Video"])
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    st.info("Status: System Running - Monitoring 5 Classes")

# 4. ส่วนหน้าจอหลัก (Main Interface)
st.title("👷 Construction Safety Monitoring System")
st.subheader("Real-time PPE Detection Dashboard")

col1, col2 = st.columns([7, 3])

with col1:
    # ส่วนที่จะแสดง Video Feed จาก YOLO
    st.markdown("### 📽️ Live Video Feed")
    st.image("https://via.placeholder.com/1280x720.png?text=Camera+Feed+Wait+for+YOLOv11", use_container_width=True)
    
    # ปุ่มควบคุมใต้ภาพ
    c1, c2, c3 = st.columns(3)
    c1.button("📸 Capture Snapshot")
    c2.button("🚨 Manual Alarm")
    c3.button("⏺️ Record Incident")

with col2:
    st.markdown("### 📊 Live Statistics")
    # แสดงตัวเลขสรุป
    st.metric(label="Total Workers", value="12", delta="2 active")
    st.metric(label="Helmet Compliance", value="92%", delta="Safe", delta_color="normal")
    st.metric(label="Critical Alerts", value="1", delta="- Violations", delta_color="inverse")
    
    st.write("---")
    st.markdown("### 🔔 Incident Logs")
    st.error("14:20:05 - Worker #5: No Safety Vest")
    st.warning("14:18:12 - Worker #2: Missing Gloves")