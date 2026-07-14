# Báo cáo kết quả — RL Trading Agent trên cổ phiếu QCOM (REL301m)

**Sinh viên:** Tạ Bảo Ngọc — **MSSV:** HE191387
**Cổ phiếu được giao:** QCOM (Qualcomm, Inc.)

## 1. Bài toán và thiết kế

Xây dựng agent Reinforcement Learning giao dịch cổ phiếu QCOM theo dữ liệu
giá đóng cửa hàng ngày, sử dụng thuật toán **Double DQN + Dueling Network +
Prioritized Experience Replay (PER)**, có **action masking** để đảm bảo agent
không vi phạm các ràng buộc nghiệp vụ (cooldown giữa hai lệnh, không mua khi
hết tiền, không bán khi hết cổ phiếu).

### Tham số cá nhân (cố định theo đề bài)

| Tham số | Giá trị |
|---|---|
| B_MIN / B_MAX (tỷ lệ tiền mặt dùng để mua) | 27% / 62% |
| SELL_MIN / SELL_MAX (tỷ lệ cổ phiếu dùng để bán) | 8% / 88% |
| TRANSACTION_FEE (phí giao dịch) | 3% |
| TRANSACTION_SESSION (cooldown) | 32 bước |
| TRANSACTION_PENALTY (chỉ dùng để shaping reward) | 2567 USD |
| TOTAL_ASSETS_INITIAL (vốn ban đầu) | 1000 USD |

### Không gian hành động
`Discrete(11)`: 0 = HOLD, 1–5 = BUY với 5 mức tỷ lệ tiền mặt (27%→62%),
6–10 = SELL với 5 mức tỷ lệ cổ phiếu (8%→88%). Giá thực thi mặc định là
**giá Close của ngày t** (`price_mode: close`).

### Không gian quan sát
Vector 95 chiều: 9 đặc trưng thị trường (log-return O/H/L/C, z-score
volume 20 ngày, SMA10/SMA20 chuẩn hoá, RSI-14, MACD histogram chuẩn hoá) ×
cửa sổ nhìn lại 10 ngày (90 chiều) + 5 đặc trưng danh mục (cash_ratio,
holding_ratio, unrealized_pnl_pct, cooldown_remaining_norm, asset_ratio).

### Reward
`reward_t = (total_assets_t − total_assets_{t−1}) / total_assets_{t−1}` trừ
đi phạt cooldown (tắt mặc định, dùng cho ablation study) và phạt "idle" khi
agent giữ HOLD quá lâu (>64 bước) mà không giao dịch. Khi tổng tài sản rơi
xuống ≤10% vốn ban đầu → kết thúc episode (phá sản) và bị phạt thêm −1.0.

### Dữ liệu
Lấy 10 năm dữ liệu OHLCV gần nhất của QCOM qua `yfinance`, chia theo thời
gian: **8 năm đầu → tập train** (2016-07-14 → 2024-07-12, 2087 phiên), **2
năm cuối → tập test out-of-sample** (2024-07-15 → 2026-07-13, 521 phiên).

## 2. Cấu hình huấn luyện

- Mạng Dueling MLP 2 lớp ẩn (256, 256), tách nhánh Value/Advantage.
- Double DQN: chọn action bằng online network, đánh giá bằng target network
  (hard update mỗi 1000 bước học).
- PER: sum-tree, α=0.6, β anneal 0.4→1.0 trong 200,000 bước.
- ε-greedy: ε giảm tuyến tính 1.0 → 0.05 trong 100,000 bước, luôn tôn trọng
  action mask (không bao giờ explore vào action không hợp lệ).
- Episode huấn luyện được chia nhỏ 252 phiên/episode (random start offset)
  từ 1956 phiên train-core (8 năm train trừ 6 tháng cuối dùng làm validation
  slice để chọn checkpoint tốt nhất).
- Tổng số bước huấn luyện: 200,000 timesteps (~103 episodes).

## 3. Kết quả huấn luyện

Đường cong reward tích lũy theo episode tăng đều đặn khi ε giảm dần:

| Episode | Bước (global step) | Cumulative reward | Total assets cuối episode | ε |
|---|---|---|---|---|
| 0 | 1,928 | 0.25 | 1,071.68 | 0.982 |
| 10 | 21,208 | 0.36 | 1,152.74 | 0.799 |
| 20 | 40,488 | 0.48 | 1,185.17 | 0.615 |
| 30 | 59,768 | 0.58 | 1,371.47 | 0.432 |
| 40 | 79,048 | 0.88 | 1,945.80 | 0.249 |
| 50 | 98,328 | 1.10 | 2,304.61 | 0.066 |
| 60 | 117,608 | 1.51 | 3,453.51 | 0.050 |
| 70 | 136,888 | 1.53 | 3,484.99 | 0.050 |
| 80 | 156,168 | 1.07 | 2,487.12 | 0.050 |
| 90 | 175,448 | 1.67 | 4,162.76 | 0.050 |
| 100 | 194,728 | 1.82 | **4,881.51** | 0.050 |

Sau khi ε chạm sàn 0.05 (~episode 55), agent khai thác chính sách đã học và
tổng tài sản cuối mỗi episode dao động trong khoảng 2,500–4,880 USD (so với
vốn ban đầu 1,000 USD), cho thấy chính sách học được có lãi rõ rệt trên dữ
liệu huấn luyện, dù có biến động giữa các episode do offset khởi đầu ngẫu
nhiên khác nhau.

Checkpoint tốt nhất (`best.pt`) được chọn theo total_assets cao nhất đạt
được trên validation slice (6 tháng cuối của tập train).

## 4. Kết quả đánh giá (greedy rollout, ε=0, có action mask)

| Split | Số phiên | Số lệnh thực thi | Cumulative reward | Total assets cuối | Score (assets/1,000,000) | Buy-and-Hold | Agent vs B&H |
|---|---|---|---|---|---|---|---|
| **Train** (2016-07-14 → 2024-07-12) | 2,087 | 60 (57 BUY, 3 SELL) | 1.6658 | **3,262.82 USD** | 0.003263 | 3,470.60 USD | −207.77 USD (**−5.99%**) |
| **Test** (2024-07-15 → 2026-07-13, out-of-sample) | 521 | 15 (15 BUY, 0 SELL) | 0.2563 | **1,071.81 USD** | 0.001072 | 831.52 USD | +240.29 USD (**+28.90%**) |

Cả hai đều xuất phát từ vốn 1,000 USD. Trên tập train, total_assets đạt đỉnh
3,660.30 USD, đáy 747.85 USD trong quá trình giao dịch. Trên tập test, đỉnh
1,462.37 USD, đáy 722.80 USD.

## 5. Nhận xét

- **Trên tập train (in-sample):** agent sinh lãi 226% (1,000 → 3,262.82 USD)
  nhưng vẫn thấp hơn chiến lược Buy-and-Hold khoảng 6%, do QCOM tăng giá khá
  mạnh và ổn định trong giai đoạn 8 năm này — chiến lược "mua và giữ" đơn
  giản đã tận dụng tối đa xu hướng tăng, trong khi agent có xu hướng thận
  trọng hơn (chỉ thực hiện 60 lệnh trong hơn 2000 phiên, phần lớn là BUY).
- **Trên tập test (out-of-sample, quan trọng nhất để đánh giá khả năng tổng
  quát hóa):** agent vượt trội Buy-and-Hold gần 29%. Nguyên nhân là giai
  đoạn này QCOM có biến động/điều chỉnh giá, khiến chiến lược mua-giữ-toàn-bộ
  chịu thiệt hại nhiều hơn, trong khi agent — nhờ cơ chế cooldown 32 bước và
  việc chia nhỏ khối lượng mua theo tỷ lệ — kiểm soát rủi ro tốt hơn và tránh
  full-exposure liên tục.
- Agent gần như không dùng lệnh SELL trên tập test (0 lệnh) — phần lớn hành
  vi là BUY dần theo tín hiệu kỹ thuật và giữ (HOLD), phù hợp với xu hướng dữ
  liệu là chủ yếu đi ngang/tăng lại vào cuối giai đoạn test.
- Action masking hoạt động đúng như thiết kế: không có lệnh BUY/SELL nào vi
  phạm cooldown 32 bước, được xác minh bằng unit test (`tests/test_action_masking.py`,
  3/3 test pass).

## 6. Kết luận

Agent Double DQN + Dueling + PER với action masking học được một chính sách
giao dịch có lãi trên cả hai tập, và đặc biệt cho thấy khả năng tổng quát
hóa tốt khi vượt trội hơn baseline Buy-and-Hold gần 29% trên dữ liệu
out-of-sample — tiêu chí quan trọng nhất để đánh giá một chiến lược giao
dịch thực tế thay vì chỉ overfit vào dữ liệu lịch sử đã thấy.

---
*Toàn bộ số liệu trong báo cáo này được sinh trực tiếp từ `train.py` (log
huấn luyện tại `outputs/logs/`, checkpoint tại `outputs/checkpoints/best.pt`)
và `evaluate.py` (nhật ký giao dịch đầy đủ tại
`outputs/trade_history_train.csv` / `outputs/trade_history_test.csv`).*
