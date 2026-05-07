import streamlit as st
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

st.set_page_config(page_title="AOI - Nhận diện Lỗi PCB", layout="wide")
st.title("Phần mềm kiểm tra lỗi bản mạch bằng YOLOv8")
st.write("Hệ thống có khả năng phân loại 6 dạng khuyết tật: Đứt mạch, Chập mạch, Lỗ thủng, Vết mẻ, Dằm đồng, Đồng rác.")

@st.cache_resource
def load_model():
    # Tự động lấy đường dẫn tuyệt đối của thư mục đang chứa file app.py này
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Nối đường dẫn thư mục đó với tệp best.pt
    model_path = os.path.join(current_dir, 'best.pt')
    
    return YOLO(model_path) # Nạp trọng số mô hình đã huấn luyện

try:
    model = load_model()
except Exception:
    st.error("Không tìm thấy tệp 'best.pt'. Hãy đảm bảo bạn đã tải mô hình về và đặt chung thư mục với code.")
    st.stop()

# 3. Bảng điều khiển (Sidebar)
st.sidebar.title("Bảng điều khiển AI")
conf_threshold = st.sidebar.slider("Ngưỡng tin cậy (Confidence)", 0.0, 1.0, 0.25, 0.05)
uploaded_file = st.sidebar.file_uploader("📥 Tải ảnh bản mạch lên", type=['jpg', 'png', 'jpeg'])

# 4. Xử lý ảnh và hiển thị kết quả
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    
    # YOLO luôn trả về kết quả dưới dạng một List đa chiều
    results = model.predict(source=image_pil, conf=conf_threshold)
    
    for result in results:
        res_image = result.plot() # Yêu cầu AI tự vẽ Bounding Box
        boxes = result.boxes # Trích xuất dữ liệu lớp lỗi
        break # Chỉ xử lý ảnh đầu tiên rồi dừng ngay lập tức
    
    # Hiển thị song song
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Ảnh Nguyên Bản**")
        st.image(image_pil, use_container_width=True)
    with col2:
        st.write("**Kết quả Phân loại từ YOLOv8**")
        # Cần đảo không gian màu BGR của YOLO sang RGB của Web
        st.image(res_image[..., ::-1], use_container_width=True)
        
    st.write("---")
    st.write("**Báo Cáo Khuyết Tật:**")
    
    if len(boxes) > 0:
        for c in boxes.cls:
            class_name = model.names[int(c)]
            st.error(f"- Phát hiện khuyết tật: **{class_name.upper()}**")
    else:
        st.success("Bản mạch ĐẠT CHUẨN, không phát hiện khuyết tật nào.")
else:
    st.info("Hệ thống đang chờ dữ liệu... Vui lòng tải ảnh lên từ thanh công cụ.")