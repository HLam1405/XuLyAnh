import streamlit as st
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
import cv2

st.set_page_config(page_title="AOI - Nhận diện Lỗi PCB", layout="wide")
st.title("Phần mềm AOI: Phát hiện & Phân tích lỗi bản mạch (YOLOv8 + OpenCV)")
st.write("""
Hệ thống kết hợp 2 giai đoạn:
1. **AI (YOLOv8):** Quét toàn bộ ảnh để định vị nhanh và phân loại 6 dạng khuyết tật (Bounding Box màu Xanh).
2. **Xử lý ảnh (OpenCV):** Phân tích hình thái học và trích xuất đặc trưng viền bên trong vùng lỗi để đo đạc chi tiết (Đường viền màu Đỏ).
""")

@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'best.pt')
    return YOLO(model_path)

try:
    model = load_model()
except Exception:
    st.error("Không tìm thấy tệp 'best.pt'. Hãy đảm bảo bạn đã tải mô hình về và đặt chung thư mục với code.")
    st.stop()

st.sidebar.title("⚙️ Bảng điều khiển")
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy YOLO (Confidence)", 0.0, 1.0, 0.25, 0.05)
uploaded_file = st.sidebar.file_uploader("📥 Tải ảnh bản mạch lên", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image_pil)
    
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    final_display_img = img_bgr.copy()
    
    cv2_report_data = []

    results = model.predict(source=image_pil, conf=conf_threshold)
    result = results[0]
    boxes = result.boxes 

    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            
            h_img, w_img = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            cv2.rectangle(final_display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(final_display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (113, 179, 60), 2)
            
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size == 0: 
                continue
            
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
            
            _, roi_thresh = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            roi_morph = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(roi_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5: 
                    cnt_shifted = cnt + np.array([x1, y1])
                    
                    cv2.drawContours(final_display_img, [cnt_shifted], -1, (0, 0, 255), 2)
                    
                    perimeter = cv2.arcLength(cnt, True)
                    
                    cv2_report_data.append({
                        "class": class_name,
                        "area": area,
                        "perimeter": perimeter
                    })

    final_rgb_display = cv2.cvtColor(final_display_img, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Ảnh Gốc")
        st.image(image_pil, use_container_width=True)
    with col2:
        st.write("### Phân Tích Hybrid (YOLO + OpenCV)")
        st.caption("🟩 Khung Xanh: YOLO định vị lỗi | 🟥 Viền Đỏ: OpenCV phân tích bề mặt")
        st.image(final_rgb_display, use_container_width=True)
        
    st.write("---")
    st.write("### 📊 Báo Cáo Phân Tích Đặc Trưng Hình Học")
    
    if len(boxes) > 0:
        st.warning(f"⚠️ Phát hiện **{len(boxes)}** vùng nghi ngờ khuyết tật trên bản mạch.")
        
        if cv2_report_data:
            for idx, data in enumerate(cv2_report_data):
                st.write(f"**Lỗi {idx + 1} - `{data['class'].upper()}`:**")
                st.write(f"- Diện tích (Area): `{data['area']:.2f}` pixels")
                st.write(f"- Chu vi viền (Perimeter): `{data['perimeter']:.2f}` pixels")
        else:
            st.info("AI phát hiện lỗi nhưng OpenCV chưa trích xuất được viền cụ thể (có thể lỗi quá mờ hoặc trùng màu nền).")
    else:
        st.success("✅ Bản mạch ĐẠT CHUẨN, không phát hiện khuyết tật nào.")
else:
    st.info("Hệ thống đang chờ dữ liệu... Vui lòng tải ảnh lên từ thanh điều khiển bên trái.")
