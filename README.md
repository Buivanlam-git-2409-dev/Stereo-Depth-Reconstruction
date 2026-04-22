# Stereo Depth Reconstruction (Tái tạo độ sâu từ ảnh Stereo)

Dự án này triển khai các thuật toán so khớp stereo (stereo matching) để tính toán bản đồ độ chênh (disparity map) từ một cặp ảnh được căn chỉnh (rectified images). Từ bản đồ độ chênh này, chúng ta có thể suy ra thông tin độ sâu của các vật thể trong không gian 3D.

## 🚀 Điểm nổi bật (Highlights)
- **Tối ưu hóa NumPy (Vectorization):** Không sử dụng vòng lặp lồng nhau (nested loops) chậm chạp, thuật toán được tối ưu để chạy nhanh hơn hàng chục lần.
- **Đa dạng thuật toán:** Triển khai từ cơ bản (Pixel-wise) đến nâng cao (Window-based) và so khớp dựa trên Cosine Similarity.
- **Cấu trúc chuyên nghiệp:** Code được module hóa theo tiêu chuẩn công nghiệp, dễ dàng mở rộng và bảo trì.

## 📁 Cấu trúc dự án
```text
Depth Information Reconstruction/
├── data/               # Chứa ảnh đầu vào (Left, Right)
├── results/            # Chứa các bản đồ độ chênh được tạo ra
├── notebooks/          # Quá trình nghiên cứu và thử nghiệm (Jupyter Notebook)
│   └── depth_infor_recons.ipynb
├── src/                # Mã nguồn chính của dự án
│   ├── matching.py     # Triển khai các thuật toán so khớp
│   └── utils.py        # Các hàm bổ trợ (xử lý ảnh, lưu file)
├── main.py             # File chạy chính với giao diện dòng lệnh (CLI)
├── requirements.txt    # Danh sách thư viện cần thiết
└── README.md           # Hướng dẫn sử dụng
```

## 🛠️ Cài đặt
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Cách sử dụng
Bạn có thể chạy dự án trực tiếp từ dòng lệnh với nhiều tùy chọn khác nhau:

### 1. So khớp theo cửa sổ (Window-based) - Khuyên dùng:
```bash
python main.py --left data/left.png --right data/right.png --method window --window-size 5 --max-disparity 64
```

### 2. So khớp theo Pixel:
```bash
python main.py --left data/left.png --right data/right.png --method pixel --metric l2
```

### 3. So khớp dựa trên Cosine Similarity:
```bash
python main.py --left data/left.png --right data/right.png --method cosine --window-size 3
```

## 🧠 Giải thích kỹ thuật
- **Pixel-wise Matching:** Tính toán sự khác biệt giữa từng pixel dựa trên khoảng cách L1 (SAD) hoặc L2 (SSD).
- **Window-based Matching:** Sử dụng một cửa sổ trượt (sliding window) để tính tổng chi phí trong một vùng lân cận, giúp giảm nhiễu đáng kể so với phương pháp pixel-wise. Dự án sử dụng `cv2.boxFilter` để tối ưu hóa việc tính toán vùng lân cận.
- **Cosine Similarity:** Đo lường góc giữa các vector đặc trưng của các vùng ảnh, giúp thuật toán hoạt động ổn định hơn trong các điều kiện ánh sáng thay đổi giữa hai camera.

## 📊 Kết quả
Kết quả sẽ được lưu trong thư mục `results/` dưới dạng ảnh xám (grayscale) và ảnh màu (color map) để dễ dàng quan sát.

---
# Stereo-Depth-Reconstruction
