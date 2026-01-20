# ISAC Deep Learning Channel Estimation (V3.6)

Dự án sử dụng Deep Learning để ước lượng kênh truyền thông trong hệ thống ISAC (Integrated Sensing and Communication) theo cơ chế **Semi-Blind One-Pass**.

**Phiên bản tốt nhất:** `V3.6 Optimized Results`
- **Method:** 4 Pilots + Ridge LS + Residual MLP
- **Test NMSE:** -40.95 dB (+36.12 dB vs GSFDIC)

## 1. Yêu cầu Hệ Thống

### Environment
Sử dụng Anaconda để quản lý môi trường:
```bash
conda create -n ai_env python=3.9
conda activate ai_env
pip install numpy torch scipy matplotlib h5py onnxruntime
```

### MATLAB
- Cần MATLAB R2023a trở lên (để chạy các script generate data và comparison).
- Cần cài đặt **Communications Toolbox** và **Phased Array System Toolbox** (tuỳ chọn).

## 2. Hướng Dẫn Chạy (Step-by-Step)

### Bước 1: Tạo Dataset (MATLAB)
Mở MATLAB, trỏ đến thư mục dự án và chạy:
```matlab
generate_dataset_v36_optimized
```
*Script sẽ tạo ra file `dataset_v36_optimized.h5` (~160MB).*

### Bước 2: Training (Python)
Chạy script training để huấn luyện mô hình:
```bash
conda run -n ai_env python training_v36_optimized.py --epochs 100
```
*Kết quả:*
- `best_v36_opt.pt`: PyTorch checkpoint tốt nhất.
- `figures_v36/`: Các biểu đồ training loss/NMSE.

### Bước 3: Export ONNX (Python)
Chuyển đổi model sang ONNX để dùng trong MATLAB:
```bash
conda run -n ai_env python export_onnx_v2.py
```
*Output:* `v36_opt.onnx`

### Bước 4: Kiểm thử và So sánh (MATLAB)
Chạy script so sánh Monte Carlo (100 trials):
```matlab
compare_v36_simple
```
*Script sẽ so sánh V3.6 với GSFDIC 4-vòng lặp và in ra kết quả NMSE.*

## 3. Cấu Trúc Dự Án

```
isac/
├── compare_v36_simple.m            # Script kiểm thử và so sánh chính
├── generate_dataset_v36_optimized.m # Script tạo dữ liệu train/val
├── training_v36_optimized.py       # Script training (PyTorch)
├── export_onnx_v2.py               # Script export ONNX
├── onnx_inference_helper.py        # Helper gọi Python từ MATLAB
├── useful_function/                # Các hàm phụ trợ MATLAB (fdic, etc.)
├── report_v3.html                  # Báo cáo chi tiết (HTML)
└── report_v3.pdf                   # Báo cáo chi tiết (PDF)
```

## 4. Kết Quả Mong Đợi

Khi chạy `compare_v36_simple`, kết quả sẽ tương tự:
```
=== V3.6 Optimized vs GSFDIC Comparison ===
Trials: 100, Pilots: 4

...

=== Results (100 trials) ===
GSFDIC mean:      -4.83 dB
V3.6 DL mean:     -40.95 dB
Improvement:      +36.12 dB
Win rate:         100.00 %
```

## 5. Lưu ý
- Nếu gặp lỗi Python trong MATLAB, hãy kiểm tra lại `pyenv` trong MATLAB để trỏ 
đúng đến env `ai_env`.
- File `.h5` và `.pt` lớn sẽ bị git ignore. Chỉ cần chạy lại Bước 1 & 2 để tái tạo.
