# Báo Cáo: Deep Learning cho Ước Lượng Kênh ISAC

**Cập nhật:** 19/01/2026

---

## 1. Tổng Quan Dự Án

### Mục tiêu
Thay thế thuật toán GSFDIC 4-vòng lặp bằng Deep Learning 1-pass cho ước lượng kênh truyền thông trong hệ thống ISAC (Integrated Sensing and Communication).

### Kết quả chính

| Version | Pilots | Val NMSE | Test NMSE | Ghi chú |
|---------|--------|----------|-----------|---------|
| **V3.6 Opt** | 4 | -41.09 dB | -45.63 dB | ✅ Best |
| V3.7 | 4 | -40.47 dB | - | Larger model |
| V3.8 | 4 | -40.47 dB | - | Transformer |
| V4 (blind) | 0 | +2.7 dB | - | ❌ Failed |
| V5 (sensing) | 4 | TBD | TBD | In progress |

---

## 2. Tài Liệu Tham Khảo

### 2.1 Sensing-Assisted Channel Estimation (ISAC)

| # | Paper Title | Source | Year | Key Idea |
|---|-------------|--------|------|----------|
| 1 | **Sensing-Assisted Channel Estimation for Bistatic OFDM ISAC Systems** | arXiv | 2025 | Dùng sensing results để cải thiện LMMSE channel estimation |
| 2 | **Deep Learning-Based Channel Estimation for IRS-Assisted ISAC** | arXiv | 2024 | Two-CNN framework cho cả sensing và communication channels |
| 3 | **Joint Sensing and Communication Channel Estimation** | IEEE | 2024 | DNN tại BS cho sensing, DNN tại UE cho communication |
| 4 | **LISAC: Deep Learning-Based Coded Waveform Design for ISAC** | arXiv | 2024 | RNN parameterize pilot/data encoding, joint training |
| 5 | **SisRafNet: OFDM Channel Estimation with Frequency Recurrence** | arXiv | 2024 | Novel RNN exploiting frequency correlation |

### 2.2 Blind/Semi-Blind Channel Estimation

| # | Paper Title | Source | Year | Key Idea |
|---|-------------|--------|------|----------|
| 6 | **Deep Learning-Based Blind Channel Estimator for OFDM** | ResearchGate | 2025 | Blind estimation không cần pilot symbols |
| 7 | **CNN-Based Resource Grouping for Blind Estimation** | MDPI | 2024 | Higher-order statistics + CNN adaptivity |
| 8 | **Hybrid CNN-BiLSTM-GRU for Channel Estimation** | IEEE | 2024 | Spatial-temporal correlations, reduced pilots |
| 9 | **Deep Image Prior for Channel Estimation** | IEEE | 2024 | Untrained DNN with limited pilots |
| 10 | **2D Image Super-Resolution for OFDM Channel** | arXiv | 2024 | Channel response as 2D image, SRCNN/DnCNN |

### 2.3 IRS và ISAC Integration

| # | Paper Title | Source | Year | Key Idea |
|---|-------------|--------|------|----------|
| 11 | **Complex-Valued CNN for IRS-ISAC Channel Estimation** | IEEE | 2024 | Preserve phase information in CSI |
| 12 | **Radar-Communication Fusion for Enhanced CSI** | Springer | 2024 | Uformer-based sensing-communication fusion |
| 13 | **LLM-Based Sensing-Assisted Channel Prediction** | arXiv | 2024 | Adapt text LLM to handle complex-matrix CSI |

---

## 3. Phương Pháp Đề Xuất

### V5: Sensing-Assisted Channel Estimation

**Ý tưởng:** Tận dụng kết quả radar detection (từ CFAR) làm additional features cho channel estimation.

**Features (227 total):**
- Spectral features: 192 (band_mean, std, max)
- Global statistics: 9
- Norm factor: 1
- h_ls (Ridge): 8
- **Sensing features: 17** (ranges, velocities, powers, residual)

**Expected improvement:** -43 to -46 dB NMSE

---

## 4. Links Paper

1. arXiv ISAC: https://arxiv.org/search/?query=ISAC+channel+estimation+deep+learning
2. IEEE Xplore: https://ieeexplore.ieee.org (search: "sensing assisted channel estimation")
3. MDPI: https://www.mdpi.com (search: "blind OFDM channel estimation CNN")
4. ResearchGate: https://www.researchgate.net

---

## 5. Kết Luận

- **V3.6 Optimized** đạt kết quả tốt nhất: -45.63 dB với chỉ 4 pilots
- **Truly blind (V4)** không khả thi với current features
- **V5 Sensing-Assisted** đang triển khai - kỳ vọng cải thiện thêm
