# 💊 Nền tảng Phân tích Thuốc Thông minh - Streamlit UI

Nền tảng phân tích thuốc thông minh sử dụng ChromaDB và AI để tìm kiếm và phân tích thuốc.

## 🚀 Khởi chạy ứng dụng

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Chạy ứng dụng

#### Phiên bản tiếng Anh:
```bash
streamlit run streamlit_app.py
```
Ứng dụng sẽ mở tại: http://localhost:8501

#### Phiên bản tiếng Việt (Khuyến nghị):
```bash
streamlit run streamlit_app_vi.py --server.port 8502
```
Ứng dụng sẽ mở tại: http://localhost:8502

## 🎯 Tính năng chính

### 1. 🔍 Semantic Search
- Tìm kiếm thuốc bằng ngôn ngữ tự nhiên
- Mô tả triệu chứng hoặc loại thuốc cần tìm
- Hiển thị độ tương đồng và thông tin chi tiết
- Tùy chọn tìm kiếm theo composition hoặc side effects

**Ví dụ tìm kiếm:**
- "pain relief headache"
- "antibiotic for infection"
- "sleep disorder medication"

### 2. 🔄 Drug Substitution
- Tìm thuốc thay thế với tác dụng tương tự
- Bộ lọc tránh tác dụng phụ mạnh
- Ưu tiên thuốc có review tốt
- Sắp xếp theo độ tương đồng hoặc điểm review

### 3. ⚠️ Side Effect Analysis
- Phân tích tương tác giữa nhiều thuốc
- Cảnh báo tác dụng phụ tiềm ẩn
- Biểu đồ phân tích theo hệ cơ quan
- Hỗ trợ multi-select thuốc

### 4. 💬 Medical Q&A Chatbot
- Hỏi đáp tự nhiên về thuốc và sức khỏe
- RAG system kết hợp ChromaDB
- Gợi ý thuốc dựa trên triệu chứng
- Chat history được lưu trữ

### 5. 🏭 Manufacturer Analytics
- Phân tích các nhà sản xuất thuốc
- Thống kê số lượng thuốc và chất lượng
- So sánh giữa các hãng
- Biểu đồ phân bố review

### 6. 📊 Dashboard Overview
- Tổng quan hệ thống
- Top manufacturers và categories
- Thống kê chất lượng thuốc
- Phân tích dữ liệu tổng thể

## 🔧 Cấu trúc dữ liệu

Ứng dụng sử dụng 4 ChromaDB collections:
- `drugs_main`: Thông tin chính về thuốc
- `drugs_side_effects`: Tác dụng phụ
- `drugs_composition`: Thành phần thuốc  
- `drugs_reviews`: Đánh giá từ người dùng

## 📱 Giao diện

- **Sidebar Navigation**: Chọn tính năng từ menu bên trái
- **Responsive Design**: Tương thích với các kích thước màn hình
- **Interactive Charts**: Biểu đồ tương tác với Plotly
- **Real-time Search**: Tìm kiếm và phân tích theo thời gian thực

## 🎨 Customization

File CSS tùy chỉnh được nhúng trong `streamlit_app.py`:
- Màu sắc theo theme y tế
- Card design cho thuốc
- Responsive layout
- Hover effects

## ⚡ Performance

- Model AI được cache với `@st.cache_resource`
- ChromaDB client persistent connection
- Optimized queries với limit
- Background processing cho các tác vụ nặng

## 🔒 Disclaimer

⚠️ **Quan trọng**: Thông tin trong ứng dụng chỉ mang tính chất tham khảo. Luôn tham khảo ý kiến bác sĩ trước khi sử dụng thuốc.

## 🐛 Troubleshooting

### Lỗi kết nối ChromaDB
```
Error connecting to ChromaDB
```
**Giải pháp**: Đảm bảo thư mục `./chroma_db` tồn tại và có dữ liệu

### Lỗi model AI
```
Error loading model
```
**Giải pháp**: Kiểm tra kết nối internet để tải model lần đầu

### Performance chậm
- Cài đặt watchdog: `pip install watchdog`
- Tăng RAM nếu có thể
- Giảm `n_results` trong search

## 📞 Hỗ trợ

Nếu gặp vấn đề, hãy kiểm tra:
1. Dependencies được cài đầy đủ
2. ChromaDB có dữ liệu
3. Kết nối internet ổn định
4. Port 8501 không bị chiếm dụng