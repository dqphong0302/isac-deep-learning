# ISAC Deep Learning Channel Estimation

Deep Learning approach for communication channel estimation in ISAC (Integrated Sensing and Communication) systems.

## 🎯 Results

| Method | NMSE | Error | Speed |
|--------|------|-------|-------|
| GSFDIC fb=4 (baseline) | -6.12 dB | 24.46% | 4 iterations |
| **Deep Learning** | **-50.05 dB** | **0.0025%** | **1-pass** |

**Improvement: ~10,000x more accurate**

## 📁 Project Structure

```
├── generate_dataset_hcom_1d_fixed_phase.m   # Dataset generation (MATLAB)
├── training_hcom1d_residual_clean_win_optimized.py  # Model training (Python)
├── export_onnx_hcom1d_smallfeat.py          # ONNX export
├── compare_dl_vs_gsfdic_accurate.m          # Comparison script (MATLAB)
├── onnx_inference_helper.py                 # MATLAB-Python bridge
├── best_hcom1d_fixed.pt                     # Trained model
├── hcom1d_fixed.onnx                        # ONNX model
├── REPORT_ISAC_DL.md                        # Detailed report
├── figures/                                  # Result plots
├── useful_function/                          # MATLAB helper functions
└── goc/                                      # Original baseline code
```

## 🚀 Hướng Dẫn Chạy Code

### Bước 1: Cài đặt môi trường

**Python (Anaconda):**
```bash
conda create -n ai_env python=3.10
conda activate ai_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install h5py onnx onnxruntime
```

**MATLAB:**
- Yêu cầu: R2023a trở lên
- Toolboxes: Communications, Signal Processing

### Bước 2: Tạo Dataset (MATLAB)

Mở MATLAB, chạy:
```matlab
cd e:\isac
addpath('./useful_function/')
generate_dataset_hcom_1d_fixed_phase
```
→ Tạo file `dataset_hcom_1d_fixed_phase.h5` (~25MB, 2000 samples)

### Bước 3: Train Model (Python)

Mở terminal/PowerShell:
```bash
cd e:\isac
conda activate ai_env
python training_hcom1d_residual_clean_win_optimized.py ^
    --h5 dataset_hcom_1d_fixed_phase.h5 ^
    --epochs 100 ^
    --batch 64 ^
    --lr 2e-3 ^
    --device cuda ^
    --ckpt best_hcom1d_fixed.pt
```
→ Tạo file `best_hcom1d_fixed.pt` (model đã train)

### Bước 4: Export ONNX (Python)

```bash
python export_onnx_hcom1d_smallfeat.py ^
    --ckpt best_hcom1d_fixed.pt ^
    --out hcom1d_fixed.onnx ^
    --opset 17
```
→ Tạo file `hcom1d_fixed.onnx` (model cho MATLAB)

### Bước 5: So sánh kết quả (MATLAB)

```matlab
compare_dl_vs_gsfdic_accurate
```
→ Hiển thị bảng so sánh và tạo 5 đồ thị trong thư mục `figures/`

### Bước 6: Xem báo cáo

Mở file `REPORT_ISAC_DL.md` để xem kết quả chi tiết.

## 📊 Key Features

- **ResidualMLP Architecture**: 3 layers, 256 width, ~760K parameters
- **Residual Learning**: `h_pred = h_ls + Δh`
- **210 Input Features**: Band statistics + global stats + LS estimate
- **NMSE Loss**: Normalized for fair comparison across power levels

## 🔧 Requirements

### Python
- PyTorch >= 2.0
- h5py
- onnx, onnxruntime

### MATLAB
- R2023a or later (for ONNX opset 17)
- Communications Toolbox
- Signal Processing Toolbox

## 📝 License

MIT License

## 📚 References

- Original baseline: `goc/ISAC_anh_Phong.m`
- Channel model: ITU-R M.1225 (Indoor Office)
- PDP: [0, -9.7, -19.2, -22.8] dB
