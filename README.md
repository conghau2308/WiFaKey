# WiFaKey – Improved Biometric Cryptosystem (Research & Service Implementation)

> **Trạng thái:** Nghiên cứu thực nghiệm | Cập nhật lần cuối: 08/2026

Đây là công trình cải tiến và tích hợp hệ thống mật mã sinh trắc học **WiFaKey** (dựa trên Fuzzy Commitment Scheme, LDPC và Neural-MS Decoder) vào kiến trúc xác thực phân tán cho thương mại điện tử.

---

## Tổng quan hệ thống

Hệ thống chuyển đổi ảnh khuôn mặt thành khóa mật mã (160 bit) mà **không lưu trữ** ảnh hay đặc trưng thô, hướng tới các yêu cầu bảo vệ thông tin sinh trắc học của ISO/IEC 24745:2022.

**Pipeline cốt lõi:**

`Ảnh → FaceProcessor (InsightFace) → AdaFace (Embedding 512-d) → AdaMTrans (Binarize + Mask κ) → LDPC (BG2, N=52, Z=16) → Neural-MS Decoder → Key`

---

## Biometric Enrollment & Verification Flow

Quy trình xử lý sinh trắc học được chia làm hai pha chính: **Enroll** (đăng ký) và **Verify** (xác thực). Mục tiêu là biến đổi đặc trưng khuôn mặt (một vector số thực) thành một **ma trận nhiễu** (helper data) mà từ đó không thể suy ngược trực tiếp đặc trưng gốc, nhưng vẫn cho phép xác thực khi so sánh với một lần quét mới.

### 1. Pha Enroll (Đăng ký)

- **Bước 1:** Trích xuất embedding 512 chiều từ ảnh khuôn mặt (AdaFace).
- **Bước 2:** Lượng tử hóa và mã hóa thermometer (`lssc_binary`) để chuyển embedding thành chuỗi bit thô dài **1536 bit**.
- **Bước 3:** Áp dụng **mặt nạ ngẫu nhiên** (tham số `κ`) để chỉ giữ lại một tập con các bit ổn định, tạo thành chuỗi `b` dài **832 bit** (khớp với độ dài codeword LDPC). *(Cần đối chiếu lại giá trị κ thực tế trong code: nếu κ = 0.3125 là tỉ lệ bit bị che trên 1536 bit thì số bit giữ lại sẽ là ~1056, không phải 832 — con số này trong bản nháp trước không khớp với phép tính.)*
- **Bước 4:** Sinh khóa bí mật `K` (160 bit) ngẫu nhiên, mã hóa bằng LDPC để tạo codeword `C` (832 bit).
- **Bước 5:** Tính **helper data** `δ = b XOR C`.
- **Bước 6:** Lưu trữ lên server:
  - `helper_data` (δ)
  - `reliability_mask` (mặt nạ ngẫu nhiên đã dùng)
  - `key_hash` = SHA256(K)
  - `service_salt` (muối cố định cho service)
  - `user_secret` (dùng để xác thực kênh an toàn)

> **Ý nghĩa bảo mật:** Server chỉ lưu δ, mặt nạ và hash — không lưu ảnh hay embedding gốc. Với một fuzzy commitment scheme có codeword đủ entropy và mask không tương quan với độ tin cậy bit, việc suy ngược embedding hay khóa K trực tiếp từ δ là không khả thi về mặt tính toán. Đây là một tính chất cần được chứng minh/kiểm chứng cho tham số cụ thể của hệ thống, không phải một đảm bảo tuyệt đối mặc nhiên.

### 2. Pha Verify (Xác thực)

- **Bước 1:** Client (native app) gửi `username` lên server, nhận về `helper_data`, `reliability_mask`, `nonce`, `user_secret`, `service_salt`.
- **Bước 2:** Client chụp ảnh mới → trích xuất embedding → áp dụng **cùng mặt nạ** đã nhận để lấy chuỗi bit mới `b'` (832 bit).
- **Bước 3:** Tính **tín hiệu nhiễu** `y = b' XOR δ = b' XOR b XOR C`. Về bản chất, `y = C XOR (b' XOR b)`, trong đó `(b' XOR b)` là sai khác giữa hai lần quét (noise).
- **Bước 4:** Client chuyển `y` thành **LLR** (Log-Likelihood Ratio) — một giá trị số thực phản ánh độ tin cậy của từng bit. Có thể là hard-BPSK (±1) hoặc soft-LLR tùy variant.
- **Bước 5:** Client gửi `llr` (832 số thực) và `key_hash` lên server (qua Java core).
- **Bước 6:** Server Python (Decoder Service) nhận `llr`, chạy Neural-MS decoder để khôi phục codeword `C'`, trích xuất 160 bit đầu làm `K'`.
- **Bước 7:** Tính `SHA256(K')` và so sánh với `key_hash` đã lưu. Trả về `success` nếu khớp.

> **Nhấn mạnh:** Toàn bộ quá trình giải mã và so hash diễn ra trên server, **client không bao giờ nhìn thấy khóa gốc K** hay giá trị hash. Điều này bảo vệ **tính bí mật của K** ngay cả khi client bị xâm phạm — nhưng **không** ngăn được việc một client bị xâm phạm tự tính `y` từ một ảnh đánh cắp rồi gửi lên server để giả mạo xác thực (client-side injection). Đây vẫn là hướng tấn công còn bỏ ngỏ của kiến trúc hiện tại, chưa được giải quyết ở thiết kế mô tả trong tài liệu này.

---

## Kiến trúc dịch vụ (Service Architecture)

Hệ thống gồm 3 thành phần chính giao tiếp qua REST API:

| Thành phần | Vai trò | Công nghệ |
| :--- | :--- | :--- |
| **Client** (Native SDK/exe) | Thu thập ảnh, xử lý tiền xử lý AI (AdaFace) cục bộ, tạo LLR | Python / C++ / Go *(cần chốt lại 1 ngôn ngữ theo bản build thực tế — không dùng packaged web app/JS-TS vì không đủ hiệu năng cho các model ML liên quan)* |
| **IdP Core** (Java) | Quản lý session, user, helper data, hash, OIDC flow, token | Spring Boot (Java) |
| **Decode Service** (Python) | Nhận LLR, chạy Neural-MS decode, so khớp key_hash | FastAPI + TensorFlow |

**Quy trình xác thực điển hình:** Native gửi `username` → Java trả `helper_data` + `nonce` → Client tính LLR → Java gửi `llr` + `key_hash` sang Python Decoder → Python trả `success` → Java cấp token.

---

## Phát hiện & Cải tiến nổi bật (Research Outcomes)

Trong quá trình phân tích và debug, nhóm đã phát hiện và khắc phục các vấn đề cốt lõi:

| Vấn đề | Mô tả | Giải pháp / Kết luận |
| :--- | :--- | :--- |
| **Bug cứng gây luôn fail** | `wifakey_handler.py` thiếu `.astype(np.float32)` trước khi feed TF placeholder, khiến verify luôn thất bại trên môi trường cụ thể. | **Đã sửa** trên code gốc. |
| **Bug Hash nghiêm trọng** | Script test tự viết dùng `.tobytes()` trên `int32` trong khi enroll hash trên `int` (mặc định `int` của numpy là `int32` trên Windows nhưng `int64` trên Linux/Mac), khiến hash lệch dù bit giống nhau. | **Đã sửa** trong harness test, đạt `FRR=2.9%, FAR=0%` cho Baseline *(số liệu tự báo cáo — nên kiểm tra lại trước khi đưa vào luận văn)*. |
| **Lỗ hổng bảo mật với Mask** | Gán `max_mag` (tin cậy cao nhất) cho bit bị mask (mask=0) vì cho rằng giá trị này "biết trước". | **Sai lầm:** gây FAR tăng vọt 39.2%. Đã thay bằng `masked_mag=1.0` (trung tính). |
| **Hiệu năng Soft-LLR** | Soft-LLR (khoảng cách tới ngưỡng) được kỳ vọng vượt Hard-BPSK nhưng thực tế **không cải thiện**. | **Kết luận:** decoder gốc tối ưu cứng cho ±1; gán LLR biến động làm nhiễu Belief Propagation. Đây là một **kết quả âm tính có giá trị** (ghi nhận cơ sở thực nghiệm cho hướng nghiên cứu tiếp theo). |

---

## Cấu trúc thư mục dự án

```
project_root/
├── wifakey_module/              # Module gốc (ĐÃ SỬA 1 BUG)
│   ├── wifakey_handler.py       # [FIXED] Đã thêm .astype(np.float32)
│   └── weights/                 # Trọng số pre-trained cho Neural-MS
│
├── research/                    # Code nghiên cứu cải tiến (không ảnh hưởng module gốc)
│   ├── modulation/              # Các variant LLR (v0 hard, v1 soft, v2 symbol-level)
│   ├── decoder/                 # Fine-tuning scripts & weights
│   └── pipeline/                # Harness A/B test
│
├── scripts/                     # Scripts chẩn đoán, trích xuất embedding
│   └── research/                # Script debug chi tiết (parity, BER phân loại, v.v.)
│
├── datasets/                    # Dataset LFW, embedding cache, và 3 tầng pairs (tune/select/final)
│
├── experiments/                 # Kịch bản chạy thử nghiệm chính thức
│
└── wifo_decode_service/         # DỊCH VỤ DECODE CHÍNH THỨC (FastAPI)
    ├── app/
    │   ├── main.py               # API endpoints (/decode, /health)
    │   ├── decoder.py            # Wrapper khởi tạo session TensorFlow
    │   └── models.py              # Pydantic schema
    ├── requirements.txt
    └── run.sh
```

---

## Hướng dẫn cài đặt & chạy (Source Code Instructions)

Hướng dẫn này tập trung vào việc chạy **Decode Service (FastAPI)** – thành phần xương sống để Java giao tiếp với Neural-MS.

### 1. Yêu cầu hệ thống
- Python 3.8 – 3.10 (TensorFlow 2.x)
- (Khuyến nghị) NVIDIA GPU + CUDA 11.x để tăng tốc, hoặc CPU (sẽ chạy chậm hơn).
- Môi trường: `conda` hoặc `venv`.

### 2. Clone và chuẩn bị thư mục
Đảm bảo thư mục dự án có cấu trúc như trên. Đặc biệt, thư mục `wifakey_module` (chứa code gốc) và `wifo_decode_service` nằm cùng cấp.

```bash
# Ví dụ cấu trúc
/path/to/project/
├── wifakey_module/      # Code gốc của tác giả (đã sửa)
└── wifo_decode_service/ # Service chính
```

### 3. Cài đặt dependencies
```bash
cd wifo_decode_service
pip install -r requirements.txt
```

**Nội dung `requirements.txt` tiêu chuẩn:**
```
fastapi
uvicorn
python-dotenv
numpy
tensorflow>=2.0
pydantic
```

### 4. Cấu hình đường dẫn (nếu cần)
Mở `app/config.py` để kiểm tra đường dẫn tới `wifakey_module`. Mặc định, service sẽ tìm thư mục này ở cấp cha:
```python
WIFAKEY_MODULE_PATH = BASE_DIR / "wifakey_module"
```

### 5. Chạy Decode Service
```bash
# Linux / Mac
bash run.sh

# Hoặc trực tiếp với uvicorn (Windows / Debug)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Sau khi chạy, API sẽ sẵn sàng tại `http://localhost:8000`.

### 6. Kiểm tra hoạt động
- **Health Check**: `curl http://localhost:8000/health` → `{"status":"ok"}`
- **API Docs**: Truy cập `http://localhost:8000/docs` để thử nghiệm Swagger UI.

### 7. Gọi API `/decode` (Dành cho Java/Backend)

**Request (POST)**:
```json
{
  "llr": [0.1, -0.5, "... 832 số thực"],
  "key_hash": "a1b2c3... (hex string, 64 ký tự)"
}
```

**Response**:
```json
{
  "success": true,
  "message": "OK"
}
```

---

## Lưu ý quan trọng cho nhà phát triển (Developer Notes)

1. **Không sửa `wifakey_module` gốc** (trừ 1 dòng bug đã xác nhận). Mọi thử nghiệm cải tiến phải nằm trong thư mục `research/` để giữ Baseline sạch.
2. **Xóa cache khi debug**: Nếu thay đổi code logic trong `research/modulation` hoặc `decoder`, hãy xóa thư mục `__pycache__` và **restart hoàn toàn Python Kernel** (đặc biệt nếu dùng Jupyter) để tránh lỗi import từ bytecode cũ.
3. **3 tầng dữ liệu (Data Tiers)**:
   - `tune`: hiệu chỉnh tham số (κ, scale factor).
   - `select`: so sánh các phiên bản (A/B test).
   - `final` (`pairs.csv` gốc LFW): **chỉ chạy DUY NHẤT 1 lần** khi đã chốt phiên bản cuối cùng.
4. **Bảo mật**: Service Python **không** giữ secret hay key nào. Nó chỉ nhận `llr` đầu vào và `key_hash` để so sánh, trả về `success`/`fail`. Nhờ đó khóa gốc `K` không bao giờ được truyền qua mạng — nhưng như đã nêu ở phần Verify, điều này không tự động ngăn được tấn công giả mạo từ một client bị xâm phạm.

---

## Trích dẫn & Tài liệu tham khảo
- WiFaKey gốc: Dong et al., *WiFaKey: Generating Cryptographic Keys from Face in the Wild*, arXiv:2407.14804.
- Tiêu chuẩn ISO/IEC 24745:2022.

---

> **Ghi nhận:** Kết quả thực nghiệm cho thấy Soft-LLR (dù mang nhiều thông tin hơn) chưa vượt qua được Baseline Hard-BPSK, do Neural-MS được tối ưu cứng cho biên độ ±1. Dự án đã ghi nhận đầy đủ cơ sở dữ liệu và logic cho hướng nghiên cứu tiếp theo (Symbol-level LLR hoặc COVQ) trong thư mục `research/`.