from ultralytics import YOLO
import cv2

# 1. โหลด Model (ตรวจสอบให้แน่ใจว่าไฟล์ best.pt อยู่ในโฟลเดอร์เดียวกัน)
model = YOLO("best.pt") 

# 2. รันการตรวจจับ
# conf=0.5 คือค่าความเชื่อมั่น (Confidence Threshold) 
# ถ้าต้องการให้แม่นยำขึ้นให้เพิ่มค่านี้ เช่น 0.5 หรือ 0.6
results = model.predict(source="AnyConv.com__6.jpg", conf=0.5, save=True)

# 3. วนลูปเพื่อแสดงผล
for r in results:
    # r.plot() จะวาดกรอบ (Bounding Box) และชื่อคลาสลงบนภาพให้โดยอัตโนมัติ
    im_array = r.plot()
    
    # แสดงหน้าต่างผลลัพธ์
    cv2.imshow("Yugi Detection Result", im_array)
    
    # รอการกดปุ่ม (กดปุ่มใดก็ได้เพื่อไปต่อหรือปิด)
    cv2.waitKey(0) 

cv2.destroyAllWindows()