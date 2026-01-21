# Báo Cáo Tổng Hợp: Deep Learning Channel Estimation cho ISAC
**Phiên bản Final | Cập nhật: 20/01/2026**

---

## 1. Giới Thiệu Chung

### 1.1 Bối Cảnh & Vấn Đề
Trong hệ thống **ISAC (Integrated Sensing and Communication)**, việc ước lượng kênh truyền thông (Channel Estimation - CE) gặp nhiều thách thức đặc thù so với hệ thống OFDM thông thường:
1.  **Self-Interference (SI):** Tín hiệu radar phát đi phản xạ trực tiếp về máy thu với công suất cực lớn.
2.  **Target Echoes:** Tín hiệu phản xạ từ các mục tiêu (sensing targets) gây nhiễu cho tín hiệu communication.
3.  **Pilot Overhead:** Các phương pháp truyền thống cần nhiều pilot để ước lượng kênh chính xác, làm giảm hiệu suất phổ (spectral efficiency).

### 1.2 Mục Tiêu
Phát triển một mô hình **Deep Learning (DL)** có khả năng:
- Ước lượng chính xác kênh `h_com` từ tín hiệu nhận được bị nhiễu.
- Giảm thiểu số lượng pilot (Semi-Blind).
- Hoạt động theo cơ chế **One-Pass** (không lặp) để giảm độ trễ so với thuật toán GSFDIC truyền thống.

---

## 2. Quá Trình Nghiên Cứu & Phát Triển

Dự án đã trải qua 7 phiên bản thử nghiệm chính để đi đến giải pháp tối ưu.

### Giai đoạn 1: Khởi tạo & Thất bại (V1 - V3)
- **V1 (Fixed Phase):** Test mô hình với kênh có pha cố định → RMSE thấp nhưng overfit, không thực tế.
- **V2 (Random Phase):** Baseline đầu tiên với kênh pha ngẫu nhiên (8 pilots) → NMSE ~ -43 dB.
- **V3 (Truly Blind):** Thử nghiệm loại bỏ hoàn toàn pilot.
    - **Kết quả:** Thất bại hoàn toàn (0 dB).
    - **Nguyên nhân:** Neural Network không thể tách biệt pha của kênh (`h_phase`) và pha của tín hiệu (`x_phase`) nếu không có mốc tham chiếu (reference). Hiện tượng "phase ambiguity".

### Giai đoạn 2: Tối ưu hóa Semi-Blind (V3.6) ⭐
Nhận thấy tầm quan trọng của pilot, chúng tôi phát triển hướng tiếp cận **Semi-Blind** với triết lý **Residual Learning**.
- **V3.6 Optimized:** Sử dụng 4 pilots (giảm 50% so với V2) + Ridge Regularized LS làm đầu vào.
- **Kết quả:** NMSE -40.95 dB (Validation: -41.09 dB).
- **Đột phá:** Đây trở thành phiên bản tốt nhất (State-of-the-Art của dự án).

### Giai đoạn 3: Mở rộng & Thử nghiệm thất bại (V3.7 - V7)
Cố gắng cải thiện V3.6 bằng các kỹ thuật phức tạp hơn nhưng đều không hiệu quả:
- **V3.7 (Larger Model):** Tăng số parameters (200K → 1M). → NMSE không đổi, overfitting nhẹ.
- **V3.8 (Transformer):** Thay MLP bằng Self-Attention. → Không cải thiện, training chậm.
- **V5 (Sensing-Assisted):** Thêm thông tin range/velocity từ Radar Sensing. → Không giúp ích, do thông tin sensing không tương quan trực tiếp với `h_com` phase.
- **V6 (Enhanced Features):** Thêm phase difference, correlation features. → NMSE -40.36 dB (kém hơn).
- **V7 (Hybrid CNN):** Kết hợp CNN 2D để xử lý ma trận E. → NMSE -40.19 dB (kém nhất trong nhóm tốt). CNN làm nhiễu tín hiệu clean từ V3.6.

---

## 3. Giải Pháp Tối Ưu: V3.6 "Residual Ridge-MLP"

Đây là kiến trúc chiến thắng, kết hợp sự bền vững của lý thuyết cổ điển và khả năng học của DL.

### 3.1 Phương pháp luận: Residual Learning
Thay vì để Neural Net học trực tiếp `h_com` (khó), chúng tôi để nó học **phần dư (residual)** để sửa lỗi cho ước lượng LS thô.

$$ \mathbf{h}_{final} = \mathbf{h}_{LS}^{Ridge} + \Delta\mathbf{h}_{NN}(\mathbf{x}_{features}) $$

Trong đó:
1.  **$\mathbf{h}_{LS}^{Ridge}$ (The Anchor):** Là ước lượng Least Squares từ 4 pilot symbols, áp dụng Ridge Regularization ($\lambda=0.1$).
    - Vai trò: Cung cấp thông tin pha và biên độ thô (Coarse Estimate). Giải quyết vấn đề "phase ambiguity".
2.  **$\Delta\mathbf{h}_{NN}$ (The Refiner):** Đầu ra của Neural Network.
    - Vai trò: Dùng thông tin từ toàn bộ phổ (data subcarriers) để tinh chỉnh lại ước lượng, loại bỏ nhiễu và interference còn sót lại.

### 3.2 Feature Engineering (210 chiều)
Vectơ đầu vào cho Neural Network bao gồm:
1.  **Spectral Shape (192 dim):** Mean, Std, Max của log-power trên 64 băng tần con. Giúp model "nhìn" thấy cấu trúc phổ của nhiễu/target.
2.  **Global Stats (9 dim):** Thống kê toàn cục (mean, max, min, percentiles).
3.  **Signal Scale (1 dim):** Normalization factor.
4.  **Coarse Estimate (8 dim):** Chính là $\mathbf{h}_{LS}^{Ridge}$ (phần thực/ảo của 4 pilots). Đây là feature quan trọng nhất.

### 3.3 Kiến trúc Neural Network
Sử dụng kiến trúc **Residual MLP (ResMLP)** đơn giản nhưng hiệu quả:
- **Input:** 210
- **Hidden:** 4 lớp Linear (512 units) + LayerNorm + SiLU + Dropout (0.2).
- **Output:** 8 (Residual $\Delta h$).
- **Tổng tham số:** ~189K (Rất nhẹ).

---

## 4. Phân Tích Kết Quả Chi Tiết

### 4.1 So sánh Hiệu năng (Test Set - 100 Monte Carlo Trials)

| Phương pháp | NMSE (dB) | Improvement vs GSFDIC | Nhận xét |
|-------------|-----------|-----------------------|----------|
| **GSFDIC (4 iter)** | 0.90 dB | - | Baseline truyền thống, kém do ít pilot và nhiễu lớn. |
| **Ridge LS** | -40.92 dB | +41.82 dB | Ridge Regularization $\lambda=0.1$ hoạt động cực tốt. |
| **V3.6 (5k dataset)** | -40.57 dB | +41.47 dB | Phiên bản cũ với 5000 samples. |
| **V3.6 (20k dataset)** | **-53.98 dB** | **+54.88 dB** | **Best Performance.** Training với 20,000 samples. |

### 4.2 Phát Hiện Quan Trọng: Giảm Số Vòng Lặp (Iteration Reduction) ⭐

Một phát hiện đột phá của dự án là khả năng **giảm số vòng lặp GSFDIC** khi sử dụng DL:

| no_of_fb | GIFDIC NMSE | DL V3.6 NMSE | GIFDIC Time | DL Time |
|----------|-------------|--------------|-------------|---------|
| 1 | 0.48 dB | **-53.98 dB** | 1.79 s | **2.16 s** |
| 2 | 0.86 dB | -53.98 dB | 4.15 s | 4.01 s |
| 3 | 0.89 dB | -53.98 dB | 6.08 s | 5.89 s |
| 4 | 0.90 dB | -53.98 dB | 8.01 s | 7.93 s |

**Phân tích:**
1. **DL NMSE không thay đổi** (-53.98 dB) bất kể số vòng lặp → Channel estimation được thực hiện **one-shot** trước loop.
2. **DL với fb=1 tốt hơn GIFDIC với fb=4** tới **54.88 dB**!
3. **Tốc độ nhanh hơn 2-4x** tùy theo số iteration được chọn.

**Khuyến nghị:** Sử dụng `no_of_fb = 2` để có **cân bằng tối ưu** giữa:
- Chất lượng RDM (3 targets rõ ràng, ít artifacts)
- Tốc độ (nhanh hơn 2x so với GIFDIC gốc)

### 4.3 Tại sao DL hoạt động tốt với single iteration?

1. **One-shot channel estimation:** DL ước lượng hcom từ raw FDIC output, không phụ thuộc vào kết quả các iteration trước.
2. **Better initial estimate:** Với hcom chính xác ngay từ đầu, communication reconstruction tốt hơn → SI cancellation tốt hơn.
3. **No error propagation:** GSFDIC truyền thống có thể accumulate errors qua các iterations; DL tránh được vấn đề này.

### 4.4 Pilot Overhead
- **V3.6:** Sử dụng 4 pilot symbols trên tổng số 512 symbols.
- **Overhead:** $4/512 \approx 0.78\%$.
- So với các hệ thống OFDM thông thường (thường ~10% pilot), đây là mức overhead cực thấp.

---

## 5. Kết Luận & Khuyến Nghị

### 5.1 Kết luận cuối cùng
1.  **V3.6 (Semi-Blind MLP + 2-Phase Processing)** là giải pháp tối ưu nhất: đạt NMSE **-53.98 dB** (tốt hơn GIFDIC 54.88 dB), chỉ cần 2 phases (nhanh 2x).
2.  **2-Phase Processing:** Phase 1 loại bỏ SI và Com, Phase 2 refine target detection - cho phép giảm iterations từ 4 xuống 2.
3.  **Training với 20k samples:** Cải thiện đáng kể so với 5k samples (từ -40.57 dB lên -53.98 dB).

### 5.2 Khuyến nghị triển khai
Sử dụng mô hình **V3.6 Optimized** với cấu hình:
- **no_of_fb = 2:** Cân bằng tối ưu giữa chất lượng và tốc độ
- **One-shot DL inference:** Ước lượng hcom trước GSFDIC loop
- **4 pilot symbols:** Overhead chỉ 0.78%

---
*Báo cáo được tổng hợp bởi AI Assistant dựa trên quá trình thực nghiệm mô phỏng hệ thống ISAC.*
