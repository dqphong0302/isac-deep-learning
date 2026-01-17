# Báo Cáo Kết Quả: Deep Learning cho Ước Lượng Kênh Truyền Thông trong Hệ Thống ISAC

**Ngày cập nhật:** 17/01/2026  
**Môi trường:** MATLAB R2023a + Python 3.10 (PyTorch 2.5.1)

---

## 1. Tổng Quan

### 1.1 Bối Cảnh

Hệ thống ISAC (Integrated Sensing and Communication) tích hợp chức năng radar và truyền thông trên cùng nền tảng WiFi 6 (802.11ax):
- **Tần số:** 5 GHz, **Subcarriers:** 1024, **Symbols:** 512

**Thách thức:** Ước lượng chính xác kênh truyền thông `h_com` trong điều kiện có nhiễu radar, SI, và noise.

### 1.2 Mục Tiêu

Thay thế thuật toán GSFDIC 4 vòng lặp bằng Deep Learning 1-pass để:
- Giảm latency
- Tăng độ chính xác
- Kiểm tra khả năng generalization

### 1.3 Kết Quả Cuối Cùng (V2)

| Phương pháp | NMSE | Sai số | Win Rate |
|-------------|------|--------|----------|
| GSFDIC fb=4 | +0.85 dB | >100% | 0% |
| **V2 DL** | **-47.77 dB** | **0.0017%** | **100%** |

**Cải thiện: ~48.6 dB (~70,000x chính xác hơn)**

---

## 2. Thử Nghiệm Các Phiên Bản

### 2.1 Tổng Quan 3 Phiên Bản

| Version | Phase | h_ls Feature | Input Dim | Test NMSE | Kết quả |
|---------|-------|--------------|-----------|-----------|---------|
| V1 | Fixed | ✅ | 210 | -50 dB | ⚠️ Overfitting risk |
| **V2** | **Random** | ✅ | 210 | **-47.77 dB** | ✅ **Tốt nhất** |
| V3 | Random | ❌ | 202 | 0 dB | ❌ Thất bại |

### 2.2 V1: Fixed Phase (Ban đầu)

**Vấn đề phát hiện:**
1. Channel dùng **fixed phase** `exp(1j*[0 -0.8 1.6 -2.6])` → Model có thể học thuộc
2. h_ls như feature → Semi-blind, không phải blind hoàn toàn
3. NMSE -50 dB quá tốt → Nghi ngờ overfitting

### 2.3 V3: Truly Blind (Thất bại)

**Thay đổi:**
- Random phase channel
- Bỏ h_ls khỏi features (202 → 202 input)
- Direct prediction (không residual)

**Kết quả:** NMSE = 0 dB (model không học được)

**Kết luận:** Spectral features alone KHÔNG đủ để blind estimation

### 2.4 V2: Semi-Blind + Random Phase (Thành công)

**Thay đổi:**
- Random phase: `2π * rand(4,1)` mỗi sample
- Giữ h_ls features (210 input)
- Residual learning

**Kết quả:**
- Train NMSE: -44.22 dB
- Val NMSE: -43.77 dB
- **Test NMSE: -47.77 dB**
- Gap train-val: ~0.5 dB (không overfitting)

---

## 3. Chi Tiết V2 (Phiên bản Cuối)

### 3.1 Dataset

```matlab
% Random phase channel (key difference)
random_phase = 2 * pi * rand(4, 1);
hcom = db2mag([0 -9.7 -19.2 -22.8]).' .* exp(1j * random_phase) .* power;
```

- **Samples:** 5000
- **Features:** 210 (192 band stats + 9 global + 1 norm + 8 h_ls)

### 3.2 Model Architecture

```python
class ResidualMLP:
    # 3 layers, 256 width, 189K parameters
    def forward(x, h_rls):
        delta = net(x)
        return h_rls + delta  # Residual learning
```

### 3.3 Training

- **Epochs:** 100
- **Best val NMSE:** -43.77 dB
- **Train-Val gap:** 0.5 dB (no overfitting)

---

## 4. Kết Quả So Sánh V2 vs GSFDIC

### 4.1 NMSE Statistics

| Metric | GSFDIC fb=4 | V2 DL |
|--------|-------------|-------|
| **Mean** | +0.85 dB | **-47.77 dB** |
| Median | +2.17 dB | -47.84 dB |
| Std Dev | 4.50 dB | 6.51 dB |
| Best | -10.92 dB | -61.45 dB |
| Worst | +5.93 dB | -30.80 dB |

### 4.2 Improvement

- **V2 DL tốt hơn:** +48.62 dB
- **Win rate:** 100% (100/100 trials)
- **p-value:** 1.05e-79 (extremely significant)

---

## 5. Phân Tích Kết Quả

### 5.1 Tại Sao V2 Thành Công?

1. **h_ls là hint quan trọng:** Cung cấp điểm khởi đầu tốt cho residual learning
2. **Random phase ngăn overfitting:** Model học features thực sự, không học thuộc
3. **Spectral features bổ trợ:** Giúp refine estimate từ h_ls

### 5.2 Tại Sao V3 Thất Bại?

1. **Spectral power không encode phase:** Channel h_com ảnh hưởng phase, không chỉ power
2. **Blind estimation quá khó:** Cần thông tin từ pilot để có reference

### 5.3 GSFDIC Baseline Kém

NMSE dương (+0.85 dB) cho thấy:
- RLS không hội tụ tốt với random phase
- 4 vòng lặp không đủ khi phase thay đổi
- Error tích lũy qua các iteration

---

## 6. Kết Luận

### 6.1 Thành Tựu

✅ **V2 DL đạt -47.77 dB** - Cải thiện ~48.6 dB so với baseline

✅ **Không overfitting** - Generalize tốt với random phase

✅ **1-pass inference** - Thay thế 4 vòng lặp GSFDIC

✅ **Statistically significant** - p < 0.001

### 6.2 Bài Học

1. **h_ls feature cần thiết** - Truly blind không khả thi với spectral features
2. **Random phase quan trọng** - Ngăn overfitting, kiểm tra generalization
3. **Residual learning hiệu quả** - Học từ LS estimate tốt hơn từ đầu

### 6.3 Hướng Phát Triển

- Thử các phase distribution khác (Rayleigh, Rice)
- Tăng số samples để cải thiện V3
- Explore semi-blind với ít pilot hơn

---

## 7. Files Chính

| File | Mô tả |
|------|-------|
| `generate_dataset_v2_random_phase.m` | Dataset V2 |
| `training_v2_random_phase.py` | Training V2 |
| `compare_v2_vs_gsfdic.m` | Comparison |
| `best_v2.pt` | Checkpoint |
| `v2.onnx` | Model cho MATLAB |
| `figures_v2/` | Đồ thị kết quả |
| `v1/` | Files V1 (archived) |
