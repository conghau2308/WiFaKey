# WiFaKey – Khóa sinh trắc học từ khuôn mặt trong môi trường tự nhiên

![Version](https://img.shields.io/badge/version-1.0-blue)
![Status](https://img.shields.io/badge/status-research-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**WiFaKey** là hệ thống xác thực sinh trắc học bằng khuôn mặt, sử dụng cơ chế **Fuzzy Commitment** kết hợp **mã sửa lỗi LDPC**. Hệ thống tạo khóa mật mã từ đặc trưng khuôn mặt mà không lưu trữ template sinh trắc gốc.

---

## Mục lục
- [Tổng quan](#tổng-quan)
- [Vấn đề nghiên cứu](#vấn-đề-nghiên-cứu)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Luồng xử lý Enroll và Verify](#luồng-xử-lý-enroll-và-verify)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Các lớp bảo vệ](#các-lớp-bảo-vệ)
- [Cách chạy hệ thống](#cách-chạy-hệ-thống)
- [Đóng góp](#đóng-góp)

---

## Tổng quan

WiFaKey biến khuôn mặt thành "mật khẩu" nhưng không bao giờ lưu trữ bức ảnh hay vector đặc trưng gốc. Thay vào đó:

- **Lúc đăng ký (Enroll)**: Hệ thống trích xuất chuỗi bit sinh trắc `b` từ khuôn mặt, kết hợp với một từ mã ngẫu nhiên `C` để tạo `helper_data = b ⊕ C`. Chỉ `helper_data` và `key_hash = SHA256(K)` được lưu trên server.
- **Lúc xác thực (Verify)**: Từ khuôn mặt mới, tính `c' = helper_data ⊕ b'`, sau đó dùng **Neural-MS Decoder** (mạng nơ-ron học sâu trên đồ thị Tanner) để sửa lỗi và khôi phục khóa `K`.

WiFaKey chuyển bài toán xác thực sinh trắc thành bài toán **giải mã kênh nhiễu**, nhằm cân bằng giữa **độ chính xác** và **tính riêng tư**.

---

## Vấn đề nghiên cứu

### 1. Lỗ hổng AND-mask
Thiết kế ban đầu dùng `mask_r` để che bit, nhưng vô tình ép bit về 0 và làm lộ codeword. Attacker có thể khôi phục khóa 100% qua khử Gauss.

**Giải pháp**: Chuyển sang **Selection-Puncturing** – chọn ngẫu nhiên 832 vị trí thật, đảm bảo OTP hoàn hảo.

### 2. Rò rỉ entropy từ Reliability Mask
Để giảm BER, cần chọn bit ổn định (Margin Selection), nhưng mask này làm lộ thông tin thống kê, giảm entropy mỗi bit từ 1.0 xuống 0.73.

**Giải pháp**: **Privacy Amplification** bằng Universal Hashing – trích xuất khóa 140-bit an toàn có thể chứng minh, entropy 0.9999 bit/bit.

### 3. Tấn công từ kẻ có ảnh + mã nguồn
Nếu attacker có ảnh khuôn mặt và biết toàn bộ thuật toán, chúng có thể tạo `b'` đúng và giải mã khóa.

**Giải pháp**: **BioHashing** (phép chiếu ngẫu nhiên dùng `User_Secret`) + **Public Salted Permutation** (hoán vị bit dùng `Service_Salt`) + **Chữ ký thiết bị** (Secure Enclave/TPM).

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│ CLIENT (iOS / Android / Windows)                           │
│                                                             │
│  Ảnh → Face Detection/Alignment → AdaFace embedding        │
│    → BioHashing (User_Secret)                              │
│    → Salted Permutation (Service_Salt)                     │
│    → Binarization (LSSC)                                   │
│    → Margin Selection (chọn 832 bit)                       │
│    → LDPC Encode (enroll) / XOR (verify)                   │
│    → Empirical LLR (tính LLR từ margin)                    │
│                                                             │
│  Gửi LLR lên server                                        │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ SERVER DECODE (FastAPI + ONNX Runtime)                     │
│                                                             │
│  Nhận LLR (832 float) + key_hash                           │
│  Chạy Neural-MS ONNX → k_prime                             │
│  SHA256(k_prime) == key_hash? → success                    │
│                                                             │
│  Không thấy dữ liệu sinh trắc thô, chỉ thấy LLR đã biến đổi │
└─────────────────────────────────────────────────────────────┘
```

### Các module chính

| Module | Chức năng |
|--------|-----------|
| `face_extraction` | Face detection, alignment, AdaFace embedding |
| `biohashing` | Phép chiếu Gaussian ngẫu nhiên, bảo vệ template |
| `salted_permutation` | Hoán vị bit, chống liên kết chéo |
| `binarization` | Thermometer code, tạo bit và margin |
| `margin_selection` | Chọn bit ổn định, giảm BER |
| `empirical_llr` | Bảng tra cứu LLR, tăng khả năng sửa lỗi |
| `ldpc_encoder` | Mã hóa LDPC trên GF(2) |
| `device_signature` | Tạo payload chuẩn cho chữ ký thiết bị |
| `secure_memory` | Khóa bộ nhớ (mlock) và xóa dữ liệu |

---

## Luồng xử lý Enroll và Verify

Phần này mô tả chi tiết các thuật toán và bước xử lý khi gọi `enroll` và `verify` trong client, cũng như cách server tham gia.

### 🔐 Enroll – Đăng ký khuôn mặt

**Mục tiêu**: Từ ảnh khuôn mặt, tạo ra `helper_data` và `key_hash` để lưu trên server mà không lộ bất kỳ thông tin sinh trắc nào.

| Bước | Thuật toán / Kỹ thuật | Mô tả |
|------|----------------------|-------|
| 1. Trích xuất embedding | Face Detection (UltraFace/RetinaFace) + Alignment + AdaFace | Ảnh thô → vector 512 chiều `v` |
| 2. BioHashing | Gaussian Random Projection với `User_Secret` | `v' = M(user_secret) @ v`, chuẩn hóa norm |
| 3. Salted Permutation | Fisher-Yates shuffle với `Service_Salt` | Tạo hoán vị `π` cho 1536 bit, áp dụng lên bits và margins |
| 4. Binarization | Thermometer Code (LSSC) | Chiếu `v'` qua `M_matrix`, so với 3 ngưỡng, tạo 1536 bit và margin |
| 5. Margin Selection | Chọn top 832 bit có margin cao nhất | `idx_sel = argpartition(-margin, 832)` → `b_sel` |
| 6. Sinh khóa ngẫu nhiên | CSPRNG | `K` 160 bit (20 bytes) |
| 7. LDPC Encode | GF(2) matrix multiplication với `G` (160×832) | `C = K @ G` |
| 8. XOR helper data | Fuzzy Commitment | `helper_data = b_sel XOR C` |
| 9. Hash khóa | SHA-256 | `key_hash = SHA256(K)` |
| 10. Lưu trữ | Chỉ gửi `helper_data`, `reliability_mask`, `key_hash`, `service_salt` lên server | Không bao giờ gửi `v`, `b_sel`, hay `K` |

### 🔓 Verify – Xác thực khuôn mặt

**Mục tiêu**: Từ ảnh khuôn mặt mới, tạo ra `LLR` để gửi lên server decode, không bao giờ gửi dữ liệu sinh trắc thô.

| Bước | Thuật toán / Kỹ thuật | Mô tả |
|------|----------------------|-------|
| 1. Trích xuất embedding | Như enroll | `v'` từ ảnh verify |
| 2. BioHashing | Giống enroll, dùng `User_Secret` từ server | `v''` |
| 3. Salted Permutation | Giống enroll, dùng `Service_Salt` | `π` |
| 4. Binarization | Giống enroll | `b_full'`, `margin'` |
| 5. Áp dụng mask | `indices_from_mask(reliability_mask)` | Lấy `b_sel'` và `margin_sel` tại các vị trí đã đăng ký |
| 6. XOR noisy codeword | `noisy = b_sel' XOR helper_data` | Tính `c'` |
| 7. Empirical LLR | Bảng tra cứu từ `reliability_lookup.npz` | `llr[i] = sign(noisy[i]) * magnitude(margin_sel[i])` |
| 8. Gửi LLR lên server | JSON gồm `llr` (832 float) | Server không nhận bit sinh trắc, chỉ nhận LLR |
| 9. Server decode | Neural-MS ONNX | `K' = Decode(llr)` |
| 10. Server kiểm tra | `SHA256(K') == key_hash` | Trả về `success` |

### Sơ đồ tuần tự (Sequence Diagram)

```
Client                          Server Decode
  │                                 │
  │── 1. POST /verify/challenge ───▶│ (trả user_secret, helper_data, mask, salt)
  │                                 │
  │── 2. Tính LLR cục bộ ───────────│
  │                                 │
  │── 3. POST /verify/complete ─────▶│ (gửi llr + key_hash? không, server tự lấy)
  │                                 │
  │                                 │── 4. Neural-MS Decode → K'
  │                                 │── 5. SHA256(K') == key_hash?
  │◀──────── 6. success ─────────────│
```

**Lưu ý**: Client không bao giờ tự so hash; server tự tính độc lập, đảm bảo không tin client.

---

## Kết quả thực nghiệm

### Hiệu năng GMR / FAR

| Cấu hình | LFW | CPLFW |
|----------|-----|-------|
| Baseline BPSK | 42% | 5% |
| + Empirical LLR | 89% | 43% |
| + Margin Selection | **97.7%** | **93.0%** |
| Margin_bpsk (an toàn tuyệt đối) | 97.4% | 77.7% (FAR 0%) |

### Thực nghiệm tấn công

| Mô hình tấn công | Kết quả |
|------------------|---------|
| Composition vs Naive (331 users) | p = 1.00 (không khác biệt) |
| Hill-Climbing Binary | 0/331 thành công (5000 lần thử) |
| Partial Permutation Leak (100%) | 0/331 thành công (10000 lần thử) |

### Bảo mật khóa sau Privacy Amplification

- Độ dài khóa: **140 bit**
- Entropy: **0.9999 bit/bit**
- An toàn trước Grover: độ phức tạp 2^70

---

## Các lớp bảo vệ

WiFaKey sử dụng **5 lớp phòng thủ độc lập**:

1. **Margin Selection + Empirical LLR** – Tối ưu hiệu năng, giảm BER.
2. **BioHashing** – Không thể đảo ngược (irreversibility).
3. **Public Salted Permutation** – Chống liên kết chéo (unlinkability).
4. **Privacy Amplification** – Khóa an toàn có thể chứng minh.
5. **Chữ ký thiết bị** – Ngăn chặn tấn công từ kẻ có ảnh và mã nguồn.

---

## Cách chạy hệ thống

### 1. Cài đặt môi trường

#### Yêu cầu
- Python ≥ 3.9
- Rust ≥ 1.70 (nếu build client)
- ONNX Runtime

#### Cài đặt FastAPI server

```bash
# Tạo môi trường ảo
python -m venv wifakey-env
source wifakey-env/bin/activate  # Trên Windows: wifakey-env\Scripts\activate

# Cài dependencies
pip install fastapi uvicorn[standard] onnxruntime numpy

# Đặt file neural_ms.onnx vào thư mục decode_server/
```

### 2. Chạy Decode Server

```bash
cd decode_server
uvicorn main:app --host 0.0.0.0 --port 8001
```

Server sẽ khởi động tại `http://localhost:8001`. Kiểm tra:

```bash
curl http://localhost:8001/health
# Output: {"status":"ok","model":"neural_ms.onnx"}
```

### 3. Build client Rust

```bash
cd wifakey-core
cargo build --release
```

Client Rust sẽ tạo thư viện động (`.dll` / `.dylib` / `.so`) để shell native gọi qua FFI hoặc UniFFI.

### 4. API Endpoints

| Endpoint | Chức năng |
|----------|-----------|
| `POST /verify` | Nhận LLR + key_hash, decode và so sánh |
| `POST /decode` | Chỉ decode, trả về key dạng hex |
| `GET /health` | Kiểm tra trạng thái |

Ví dụ gọi:

```python
import requests

response = requests.post(
    "http://localhost:8001/verify",
    json={
        "llr": [0.1, -0.2, ..., 0.5],  # 832 phần tử
        "key_hash": "a1b2c3..."
    }
)
print(response.json())  # {"success": true}
```

---

## Đóng góp

Công trình này là kết quả của quá trình nghiên cứu và phát triển độc lập. Mọi đóng góp, câu hỏi hay thảo luận đều được chào đón.

**Tác giả**: Võ Công Hậu
