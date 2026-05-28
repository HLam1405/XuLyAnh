import streamlit as st
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO
import cv2

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="AOI - Nhận diện Lỗi PCB", layout="wide")
st.title("Phần mềm AOI: Phát hiện & Phân tích lỗi bản mạch")
st.write("Hệ thống kết hợp YOLOv8 (Định vị khuyết tật) và OpenCV (Đo đạc hình thái học).")

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH YOLOv8
# ==========================================
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

# ==========================================
# 3. BẢNG ĐIỀU KHIỂN (SIDEBAR) - HỖ TRỢ THƯ MỤC CỤC BỘ
# ==========================================
st.sidebar.title("⚙️ Bảng điều khiển")

# Cấu hình ngưỡng tin cậy (Confidence Threshold)
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy YOLO (Confidence)", 0.0, 1.0, 0.25, 0.05)

# [NÂNG CẤP LỚN] Lựa chọn phương thức nhập dữ liệu tránh phải Ctrl+A hoặc xóa thủ công
input_method = st.sidebar.radio("Phương thức nạp ảnh bản mạch:", ["Nhập đường dẫn thư mục cục bộ", "Tải tệp trực tiếp lên (Ctrl+A)"])

uploaded_files = []

if input_method == "Tải tệp trực tiếp lên (Ctrl+A)":
    uploaded_files = st.sidebar.file_uploader("📥 Chọn danh sách tệp ảnh", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
else:
    # Người dùng chỉ cần dán đường dẫn thư mục máy tính vào đây
    folder_path = st.sidebar.text_input("📁 Nhập đường dẫn thư mục chứa ảnh (Ví dụ: D:/PCB_Dataset):", "")
    if folder_path and os.path.isdir(folder_path):
        valid_extensions = ('.jpg', '.jpeg', '.png')
        files_in_dir = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
        
        # Tạo class giả lập cấu trúc của Streamlit file_uploader để tái sử dụng toàn bộ luồng xử lý bên dưới
        class LocalImageFile:
            def __init__(self, full_path, file_name):
                self.path = full_path
                self.name = file_name
                
        uploaded_files = [LocalImageFile(os.path.join(folder_path, f), f) for f in files_in_dir]
        if uploaded_files:
            st.sidebar.success(f"✅ Tự động kết nối thành công thư mục. Tìm thấy {len(uploaded_files)} ảnh.")
        else:
            st.sidebar.warning("Thư mục trống hoặc không chứa file ảnh hợp lệ (.jpg, .png).")
    elif folder_path:
        st.sidebar.error("❌ Đường dẫn thư mục không hợp lệ hoặc không tồn tại.")

# ==========================================
# 4. LUỒNG XỬ LÝ CHÍNH
# ==========================================
if uploaded_files:
    tab1, tab2 = st.tabs(["🔍 Kiểm tra Chi tiết (Từng ảnh)", "📊 Đánh giá Độ chính xác & Thư mục"])
    
    # ---------------------------------------------------------
    # TAB 1: XỬ LÝ CHI TIẾT 1 ẢNH 
    # ---------------------------------------------------------
    with tab1:
        selected_filename = st.selectbox("Chọn ảnh trong danh sách để phân tích:", [f.name for f in uploaded_files])
        selected_file = next(f for f in uploaded_files if f.name == selected_filename)
        
        # Đọc ảnh linh hoạt dựa trên phương thức đầu vào
        if input_method == "Tải tệp trực tiếp lên (Ctrl+A)":
            image_pil = Image.open(selected_file).convert('RGB')
        else:
            image_pil = Image.open(selected_file.path).convert('RGB')
            
        image_np = np.array(image_pil)
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        final_display_img = img_bgr.copy()
        cv2_report_data = []

        # YOLOv8 Predict
        results = model.predict(source=image_pil, conf=conf_threshold)
        boxes = results[0].boxes 

        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                
                # Giới hạn biên tọa độ
                h_img, w_img = img_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                # Vẽ Khung YOLO (Khung bao xanh lá)
                cv2.rectangle(final_display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ nhãn tên lỗi (Text màu xanh lục mượt dịu mắt, có hộp nền trắng)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(final_display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (113, 179, 60), 2)
                
                # XỬ LÝ TRÍCH XUẤT OPENCV (ROI)
                roi = img_bgr[y1:y2, x1:x2]
                if roi.size == 0: continue
                
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_blur = cv2.GaussianBlur(roi_gray, (3, 3), 0)
                _, roi_thresh = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                roi_morph = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(roi_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                box_total_area = 0
                box_total_perimeter = 0
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 5: 
                        cnt_shifted = cnt + np.array([x1, y1])
                        cv2.drawContours(final_display_img, [cnt_shifted], -1, (0, 0, 255), 1)
                        box_total_area += area
                        box_total_perimeter += cv2.arcLength(cnt, True)
                
                # [NÂNG CẤP] Lưu thêm tham số độ tin cậy 'conf' vào báo cáo chi tiết
                cv2_report_data.append({
                    "class": class_name,
                    "conf": conf,
                    "area": box_total_area,
                    "perimeter": box_total_perimeter
                })

        final_rgb_display = cv2.cvtColor(final_display_img, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Ảnh Gốc")
            st.image(image_pil, use_container_width=True)
        with col2:
            st.write("### Phân Tích Hybrid (YOLO + OpenCV)")
            st.image(final_rgb_display, use_container_width=True)
            
        st.write("---")
        st.write("### 📊 Báo Cáo Phân Tích Đặc Trưng Hình Học")
        
        if len(boxes) > 0:
            st.error(f"⚠️ Hệ thống phát hiện **{len(boxes)}** khuyết tật trên bề mặt linh kiện:")
            # [NÂNG CẤP] Hiển thị rõ ràng điểm tin cậy Confidence của từng lỗi cụ thể
            for idx, data in enumerate(cv2_report_data):
                st.write(f"**Khuyết tật {idx + 1} - `{data['class'].upper()}` (Độ tin cậy: `{data['conf'] * 100:.1f}%`):** Diện tích: `{data['area']:.2f}` px | Chu vi: `{data['perimeter']:.2f}` px")
        else:
            st.success("✅ Bản mạch sạch hoàn toàn - ĐẠT CHUẨN.")

    # ---------------------------------------------------------
    # TAB 2: ĐÁNH GIÁ THƯ MỤC VÀ TỶ LỆ CHÍNH XÁC HÀNG LOẠT
    # ---------------------------------------------------------
    with tab2:
        st.write("### 📈 Thống Kê & Phân Tích Toàn Thư Mục")
        
        if st.button("Bắt đầu Đánh giá Hàng loạt", type="primary"):
            progress_bar = st.progress(0)
            
            total_images = len(uploaded_files)
            pass_count = 0
            fail_count = 0
            total_defects = 0
            conf_scores = []
            
            for i, file in enumerate(uploaded_files):
                if input_method == "Tải tệp trực tiếp lên (Ctrl+A)":
                    img_batch = Image.open(file).convert('RGB')
                else:
                    img_batch = Image.open(file.path).convert('RGB')
                    
                res = model.predict(source=img_batch, conf=conf_threshold, verbose=False)
                bboxes = res[0].boxes
                
                if len(bboxes) == 0:
                    pass_count += 1
                else:
                    fail_count += 1
                    total_defects += len(bboxes)
                    for conf_val in bboxes.conf:
                        conf_scores.append(float(conf_val))
                
                progress_bar.progress((i + 1) / total_images)
            
            st.success("Đã hoàn tất phân tích toàn bộ thư mục dữ liệu!")
            
            avg_conf = (sum(conf_scores) / len(conf_scores)) * 100 if conf_scores else 0
            yield_rate = (pass_count / total_images) * 100
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Tổng số linh kiện", f"{total_images} ảnh")
            col_m2.metric("Tỷ lệ ĐẠT (Yield Rate)", f"{yield_rate:.1f}%")
            col_m3.metric("Tổng lỗi phát hiện", f"{total_defects} lỗi")
            col_m4.metric("Độ chính xác AI (Avg Conf)", f"{avg_conf:.1f}%")

else:
    st.info("Hệ thống đang chờ dữ liệu... Vui lòng nạp thư mục hoặc chọn ảnh từ thanh điều khiển bên trái.")
