# ISAC Deep Learning Channel Estimation - V3.6

## Tổng quan
Phiên bản V3.6 sử dụng Deep Learning để ước lượng kênh truyền thông (hcom) cho hệ thống ISAC, đạt NMSE **-53.98 dB** - tốt hơn GIFDIC truyền thống **54.88 dB**.

## Files chính

| File | Mô tả |
|------|-------|
| `generate_dataset_v36_optimized.m` | Tạo dataset training (MATLAB) |
| `training_v36_optimized.py` | Training model (Python) |
| `export_onnx_v2.py` | Export model sang ONNX |
| `isac_with_dl_v36.m` | Script DL-ISAC chính |
| `onnx_inference_helper.py` | Python helper cho ONNX |

## Quy trình chạy

### 1. Tạo Dataset (chỉ cần chạy 1 lần)
```matlab
% MATLAB - tạo 20,000 samples (~40 phút)
>> generate_dataset_v36_optimized
```
Output: `dataset_v36_optimized.h5`

### 2. Training Model (chỉ cần chạy 1 lần)
```bash
conda activate ai_env
python training_v36_optimized.py --h5 dataset_v36_optimized.h5 --epochs 100
```
Output: `best_v36_opt.pt`

### 3. Export ONNX (chỉ cần chạy 1 lần)
```bash
python export_onnx_v2.py --ckpt best_v36_opt.pt --out v36_opt.onnx
```
Output: `v36_opt.onnx`

### 4. Chạy DL-ISAC
```matlab
% Chạy script DL V3.6
>> isac_with_dl_v36
```

## Kết quả

| Method | NMSE (dB) | Iterations | Speedup |
|--------|-----------|------------|---------|
| GIFDIC (Original) | 0.90 | 4 | - |
| **DL V3.6** | **-53.98** | **2** | **2x** |

## Yêu cầu

### MATLAB
- Signal Processing Toolbox
- Communications Toolbox
- Python integration (pyenv)

### Python
- Python 3.8+
- PyTorch
- ONNX Runtime
- h5py
- numpy

## Cấu trúc thư mục
```
isac/
├── goc/                    # Script gốc (tham khảo)
│   └── ISAC_anh_Phong.m
├── useful_function/        # Helper functions
├── isac_with_dl_v36.m      # ★ Main DL script
├── generate_dataset_v36_optimized.m
├── training_v36_optimized.py
├── export_onnx_v2.py
├── onnx_inference_helper.py
├── v36_opt.onnx            # Trained model
├── best_v36_opt.pt         # PyTorch checkpoint
├── dataset_v36_optimized.h5
└── report_v3.md            # Báo cáo chi tiết
```
