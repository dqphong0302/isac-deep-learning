# Báo Cáo: Deep Learning cho Ước Lượng Kênh ISAC

**Cập nhật:** 19/01/2026

---

## 1. Tổng Quan

### Mục tiêu
Thay thế thuật toán GSFDIC 4-vòng lặp bằng Deep Learning 1-pass cho ước lượng kênh truyền thông trong hệ thống ISAC.

### Kết quả chính

| Version | Pilots | Test NMSE | So với GSFDIC |
|---------|--------|-----------|---------------|
| **V2** | 8 | **-47.77 dB** | +48.6 dB |
| **V3.6 Opt** | 4 | **-45.63 dB** | +47.1 dB |
| V3.5 | 2 | +22 dB | ❌ Failed |
| V3 (blind) | 0 | 0 dB | ❌ Failed |

---

## 2. Các Phiên Bản

### V2: Semi-Blind (8 Pilots) - Baseline
- **Input:** 210 features (192 spectral + 9 global + 1 norm + 8 h_ls)
- **Model:** ResidualMLP (256 width, 3 depth)
- **Result:** -47.77 dB, 100% win rate

### V3.6 Optimized: Semi-Blind (4 Pilots)
- **Tối ưu 1:** Ridge Regularization (λ=0.1) cho LS estimation
- **Tối ưu 2:** Auxiliary Magnitude Loss
- **Tối ưu 3:** Dropout 0.2
- **Model:** ResidualMLP (512 width, 4 depth)
- **Result:** -45.63 dB, 100% win rate

### V3/V3.5: Thất bại
- V3 (truly blind, 0 pilots): 0 dB - Không học được
- V3.5 (2 pilots): +22 dB - h_ls quá noisy

---

## 3. Kết Luận

1. **h_ls là feature quan trọng** - Truly blind không khả thi
2. **Ridge LS hiệu quả** - Giúp giảm pilot từ 8 → 4 với chỉ ~2 dB loss
3. **Trade-off tốt:** 50% pilot reduction, 95% performance retained
4. **DL 1-pass >> GSFDIC 4-loop** - Cải thiện ~47 dB

---

## 4. Files Chính

| Folder | Files |
|--------|-------|
| `v2/` | V2 (8 pilots) - Baseline |
| `v36/` | V3.6 Optimized (4 pilots) |
| `v1/` | V1 (fixed phase) - Archived |
