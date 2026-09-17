# Hướng dẫn Cài đặt và Chạy mô hình (Setup Guide)

Tài liệu này hướng dẫn chi tiết cách tải dữ liệu, thiết lập môi trường và chạy các tập lệnh cho dự án.

## 1. Tải và Tổ chức Dữ liệu (Dataset Preparation)

### Bước 1.1: Tải dữ liệu từ Google Drive
Truy cập vào liên kết Google Drive được cung cấp:
[🔗 Google Drive - Nộp bài VPR](https://drive.google.com/drive/u/1/folders/16aocU19AbLmojis5A9A-mh6jIY2we0uW)

- Đi tới thư mục **`dataset`**.
- Tải về các bộ dữ liệu tương ứng (ví dụ: `pitts`, `baidu_test`).

### Bước 1.2: Tổ chức thư mục `workdir_data`
Sau khi tải dữ liệu về, hãy giải nén và sắp xếp chúng vào một thư mục gốc gọi là `workdir_data` (bạn có thể đặt ở đâu tùy thích, nhưng cần trỏ đúng đường dẫn trong file config).

Cấu trúc thư mục phải tuân thủ chính xác như sau để các đoạn mã có thể đọc được (tương ứng với cấu hình trong `place_rec_global_config.py`):

```text
workdir_data/
├── baidu_test 
│   ├── training_images_undistort   # Ảnh database/reference
│   ├── query_images_undistort      # Ảnh query
│   └── out                         # Thư mục hệ thống sẽ tự sinh ra để lưu kết quả và features
├── pitts 
│   ├── ref                         # Ảnh database/reference
│   ├── query                       # Ảnh query
│   └── out                         # Thư mục hệ thống tự sinh
```

### Bước 1.3: Cập nhật đường dẫn Configuration
Mở file `place_rec_global_config.py` và sửa biến `workdir_data` ở đầu file để trỏ tới thư mục bạn vừa tạo ở Bước 1.2.

Ví dụ:
```python
workdir_data = 'E:/University/Year_3/Sem3/CV_InformationRetrieval/Course_Project/Datasets'
```

### Bước 1.4: Tải mô hình đã huấn luyện (Models)
Tại cùng link Google Drive trên, hãy tải toàn bộ thư mục **`models`** và đặt nó bên trong thư mục `workdir_data` (ngang hàng với `baidu_test` và `pitts`). 

Cấu trúc sau khi tải xong:
```text
workdir_data/
├── baidu_test/
├── pitts/
└── models/
    ├── finetune_mixed_data/
    │   └── (chứa file model của epoch 4)
    └── segment-anything/
```

---

## 2. Thiết lập Môi trường (Environment Setup)

Đảm bảo bạn đã cài đặt các thư viện cần thiết. Sử dụng Conda để tạo môi trường (như môi trường `segvlad` hiện tại):

```bash
conda create -n segvlad python=3.8
conda activate segvlad
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
*(Lưu ý: Bạn có thể cần điều chỉnh phiên bản CUDA cho phù hợp với phần cứng).*

---
Chi tiết về phần thiết lập này có thể được xem trong file README.md của dự án.

## 3. Chạy Mô hình (Inference)

Hệ thống cung cấp các shell scripts có sẵn trong thư mục `scripts/` để bạn có thể chạy toàn bộ pipeline (rút trích đặc trưng DINO, phân đoạn bằng SAM, sinh PCA, và chạy matching).

Để chạy đánh giá cho một bộ dữ liệu, mở Terminal (hoặc Git Bash/Zsh) và chạy:

**1. Đối với mô hình DINO Pretrained gốc:**
```bash
bash scripts/full_infer_baidu_pretrained.sh
bash scripts/full_infer_pitts_pretrained.sh
```

**2. Đối với mô hình DINO Finetuned:**
```bash
bash scripts/full_infer_baidu_finetuned.sh
bash scripts/full_infer_pitts_finetuned.sh
```

### Chú ý về tính năng IDF Weighting:
Nếu bạn muốn sử dụng tính năng đánh trọng số dựa trên IDF (Inverse Document Frequency) cho các phân đoạn đặc trưng hiếm, bạn **phải tính toán trọng số IDF trước**.

**Bước 1: Tính toán trọng số IDF**
Chạy script `compute_idf_weights.py`. Ví dụ:
- Cho mô hình gốc: `python compute_idf_weights.py --dataset pitts --vocab-vlad domain`
- Cho mô hình finetune (cần cờ `--finetuned`): `python compute_idf_weights.py --dataset pitts --vocab-vlad domain --finetuned`

**Bước 2: Sử dụng trọng số trong Pipeline**
Sau khi tính toán xong, hãy mở file script (`.sh`) và thêm cờ `--use-idf` vào dòng lệnh chạy file `place_rec_main.py` hoặc `place_rec_main_finetuned.py`.

Ví dụ:
```bash
python place_rec_main_finetuned.py --dataset pitts --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain --save-result --use-idf
```
