# WiFaKey – Improved Biometric Cryptosystem

> Trạng thái: Nghiên cứu thực nghiệm  
> Cập nhật lần cuối: 08/2026

Đây là dự án nghiên cứu và tích hợp hệ thống mật mã sinh trắc học WiFaKey, dựa trên Fuzzy Commitment Scheme, LDPC và Neural-MS Decoder, vào kiến trúc xác thực phân tán cho hệ thống thương mại điện tử.

## 1. Tổng quan hệ thống

Hệ thống sử dụng ảnh khuôn mặt để tạo khóa mật mã mà không cần lưu trữ ảnh hoặc embedding khuôn mặt dạng thô trong quá trình xác thực.

Pipeline nghiên cứu chính:

```text
Ảnh
  → Face Processor
  → AdaFace (512-d embedding)
  → AdaMTrans (binarization + mask)
  → LDPC
  → Neural-MS Decoder
  → Key
```

Các thành phần và tham số cụ thể có thể thay đổi tùy theo phiên bản thực nghiệm.

## 2. Kiến trúc dịch vụ

Hệ thống được tổ chức thành ba thành phần chính:

| Thành phần | Vai trò | Công nghệ |
|---|---|---|
| Client | Thu thập ảnh và thực hiện các bước xử lý sinh trắc học phía client | C# / WebAssembly |
| IdP Core | Quản lý user, session, helper data, hash, OIDC và token | Spring Boot / Java |
| Decode Service | Nhận dữ liệu LLR và thực hiện Neural-MS decoding | FastAPI / Python / TensorFlow |

Một luồng xác thực điển hình:

```text
Client
  → IdP Core: username
  ← IdP Core: helper_data + nonce
  → Client: tính embedding và LLR
  → IdP Core: llr + key_hash
  → Decode Service: decode
  ← Decode Service: success / fail
  → IdP Core: cấp token nếu xác thực thành công
```

## 3. Kết quả và phát hiện trong quá trình nghiên cứu

### 3.1. Lỗi kiểu dữ liệu khi sử dụng TensorFlow

Trong quá trình kiểm thử, đã xác định một trường hợp dữ liệu đầu vào cần được chuyển sang `np.float32` trước khi đưa vào TensorFlow.

Thay đổi đã thực hiện:

```python
.astype(np.float32)
```

Điều này giúp tránh lỗi không tương thích kiểu dữ liệu trong môi trường thực thi cụ thể.

> Lưu ý: Không nên khẳng định đây là nguyên nhân duy nhất khiến toàn bộ quá trình verify luôn thất bại nếu chưa tái hiện được lỗi trên tất cả môi trường.

### 3.2. Sai khác khi tính hash trong test harness

Trong quá trình xây dựng test harness, phát hiện việc chuyển đổi kiểu dữ liệu trước khi gọi `.tobytes()` có thể làm thay đổi byte representation và dẫn tới hash khác nhau dù các giá trị số có cùng ý nghĩa.

Vì vậy, quá trình tính hash trong enrollment và verification phải sử dụng cùng kiểu dữ liệu và cùng cách biểu diễn byte.

Baseline sau khi sửa test harness đạt kết quả thực nghiệm:

- FRR: 2.9%
- FAR: 0%

Các kết quả trên chỉ có ý nghĩa đối với tập dữ liệu, protocol và cấu hình thử nghiệm tương ứng; không nên xem đây là đặc tính tổng quát của hệ thống.

### 3.3. Ảnh hưởng của mask

Một thử nghiệm đã gán độ tin cậy lớn nhất (`max_mag`) cho các bit bị mask (`mask=0`) với giả định rằng các bit này có thể được xử lý như những bit có độ tin cậy cao.

Kết quả cho thấy cách làm này làm FAR tăng đáng kể, khoảng 39.2% trong cấu hình thử nghiệm tương ứng.

Sau đó, các bit bị mask được xử lý bằng một giá trị trung tính:

```text
masked_mag = 1.0
```

Kết quả này cho thấy việc xử lý thông tin bị mask cần được đánh giá riêng thay vì mặc định xem các bit đó là những bit có độ tin cậy cao.

### 3.4. Thử nghiệm Soft-LLR

Một hướng nghiên cứu là thay thế Hard-BPSK bằng Soft-LLR, trong đó độ lớn LLR được điều chỉnh dựa trên khoảng cách của giá trị đầu vào tới ngưỡng lượng tử.

Trong các thí nghiệm hiện tại, Soft-LLR chưa cho thấy cải thiện rõ ràng so với Baseline Hard-BPSK.

Một giả thuyết được đặt ra là decoder Neural-MS hiện tại được huấn luyện và điều chỉnh cho phân phối đầu vào cụ thể, vì vậy việc đưa vào LLR có độ lớn biến thiên có thể làm thay đổi phân phối mà decoder đã học.

Đây là một kết quả thực nghiệm âm tính hữu ích cho việc định hướng nghiên cứu tiếp theo. Tuy nhiên, chưa nên kết luận rằng Soft-LLR nói chung không hiệu quả.

## 4. Cấu trúc thư mục

```text
project_root/
├── wifakey_module/
│   ├── wifakey_handler.py
│   └── weights/
│
├── research/
│   ├── modulation/
│   ├── decoder/
│   └── pipeline/
│
├── scripts/
│   └── research/
│
├── datasets/
│
├── experiments/
│
└── wifo_decode_service/
    ├── app/
    │   ├── main.py
    │   ├── decoder.py
    │   └── models.py
    ├── requirements.txt
    └── run.sh
```

## 5. Cài đặt và chạy Decode Service

Phần này hướng dẫn chạy `wifo_decode_service`, thành phần cung cấp API để thực hiện Neural-MS decoding.

### 5.1. Yêu cầu

Môi trường Python cần tương thích với phiên bản TensorFlow được sử dụng trong project.

Khuyến nghị sử dụng một virtual environment riêng:

```bash
conda create -n wifo-decode python=3.9
conda activate wifo-decode
```

Hoặc sử dụng `venv`:

```bash
python -m venv .venv
```

Trên Windows:

```bash
.venv\Scripts\activate
```

Trên Linux/macOS:

```bash
source .venv/bin/activate
```

Nếu sử dụng GPU, cần cài đặt phiên bản TensorFlow và các thành phần CUDA/cuDNN tương thích với môi trường thực tế. Không nên mặc định rằng mọi phiên bản TensorFlow 2.x đều tương thích với cùng một phiên bản CUDA.

### 5.2. Cấu trúc thư mục

`wifakey_module` và `wifo_decode_service` cần nằm trong cấu trúc mà code cấu hình đang sử dụng. Ví dụ:

```text
project/
├── wifakey_module/
└── wifo_decode_service/
```

### 5.3. Cài dependencies

```bash
cd wifo_decode_service
pip install -r requirements.txt
```

Ví dụ các dependency chính:

```text
fastapi
uvicorn
python-dotenv
numpy
tensorflow
pydantic
```

Nên ưu tiên phiên bản đã được kiểm thử cùng project thay vì sử dụng `tensorflow>=2.0` một cách không giới hạn, vì TensorFlow có yêu cầu phụ thuộc vào Python, CUDA và các package khác.

### 5.4. Kiểm tra đường dẫn module

Nếu project có file `app/config.py`, kiểm tra biến đường dẫn tới module WiFaKey.

Ví dụ:

```python
WIFAKEY_MODULE_PATH = BASE_DIR / "wifakey_module"
```

Đường dẫn thực tế phải phù hợp với cấu trúc thư mục của project.

### 5.5. Chạy service

Trên Linux/macOS nếu `run.sh` đã được cấu hình:

```bash
bash run.sh
```

Hoặc chạy trực tiếp bằng Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Trong quá trình phát triển có thể sử dụng:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Service mặc định chạy tại:

```text
http://localhost:8000
```

## 6. Kiểm tra service

### Health check

Nếu endpoint `/health` được triển khai:

```bash
curl http://localhost:8000/health
```

Response dự kiến có dạng:

```json
{
  "status": "ok"
}
```

### API documentation

FastAPI thường cung cấp Swagger UI tại:

```text
http://localhost:8000/docs
```

Địa chỉ này chỉ hoạt động nếu ứng dụng FastAPI được cấu hình với documentation mặc định.

## 7. API `/decode`

Endpoint `/decode` được sử dụng để gửi dữ liệu cần decoding tới Python service.

Ví dụ request:

```json
{
  "llr": [0.1, -0.5, "..."],
  "key_hash": "a1b2c3..."
}
```

Độ dài của `llr`, format của `key_hash` và các trường bắt buộc phải tuân theo schema được định nghĩa trong `app/models.py`.

Không nên hard-code rằng `llr` luôn có 832 phần tử hoặc `key_hash` luôn có 64 ký tự nếu các giá trị này không được đảm bảo bởi implementation hiện tại.

Ví dụ response:

```json
{
  "success": true,
  "message": "OK"
}
```

Ý nghĩa cụ thể của `success` phụ thuộc vào implementation của `decoder.py` và logic kiểm tra hash.

## 8. Developer Notes

### 8.1. Giữ Baseline độc lập với code nghiên cứu

Code Baseline nên được giữ ổn định để có thể tái lập kết quả so sánh.

Các cải tiến và biến thể thử nghiệm nên được đặt trong:

```text
research/
```

Nếu cần sửa code trong `wifakey_module`, phải ghi rõ thay đổi và lý do để tránh làm thay đổi Baseline một cách không kiểm soát.

### 8.2. Cache Python

Khi thay đổi module trong quá trình nghiên cứu, đặc biệt khi làm việc với Jupyter Notebook, có thể cần restart kernel để bảo đảm module được import lại từ source mới.

Có thể xóa cache Python bằng cách xóa các thư mục:

```text
__pycache__/
```

Việc xóa cache không phải lúc nào cũng cần thiết; vấn đề thường liên quan đến module đã được import và giữ trong memory của Python process.

### 8.3. Data tiers

Dataset nghiên cứu được chia thành các tầng với mục đích khác nhau:

- `tune`: hiệu chỉnh tham số.
- `select`: so sánh các variant.
- `final`: đánh giá phiên bản đã được lựa chọn trên tập đánh giá cuối cùng.

Không nên sử dụng tập `final` để tiếp tục điều chỉnh tham số sau khi đã xem kết quả, vì điều này có thể làm sai lệch đánh giá cuối cùng.

Nếu project định nghĩa `pairs.csv` của LFW là tập `final`, cần giữ nguyên protocol đã được xác định trước khi thực hiện đánh giá.

### 8.4. Bảo mật của Decode Service

Theo kiến trúc hiện tại, Decode Service không cần lưu khóa bí mật gốc của người dùng. Service nhận dữ liệu cần thiết cho quá trình decoding và thực hiện kiểm tra theo logic được triển khai.

Tuy nhiên, việc service không lưu khóa không đồng nghĩa API tự động an toàn. Trong triển khai thực tế cần bảo vệ:

- Authentication và authorization giữa Java và Python service.
- TLS khi truyền dữ liệu qua mạng.
- Quyền truy cập endpoint.
- Logging và việc tránh ghi dữ liệu sinh trắc học hoặc dữ liệu xác thực nhạy cảm vào log.
- Rate limiting và chống replay nếu protocol yêu cầu.
- Validation đối với dữ liệu `llr` đầu vào.

## 9. Hướng nghiên cứu tiếp theo

Các kết quả hiện tại gợi ý một số hướng tiếp tục nghiên cứu:

1. Fine-tune Neural-MS Decoder với phân phối Soft-LLR mới.
2. Nghiên cứu Symbol-level LLR.
3. Khảo sát COVQ hoặc các phương pháp lượng tử hóa khác.
4. Đánh giá ảnh hưởng của scale và calibration của LLR.
5. Kiểm tra robustness trên các điều kiện và dataset khác nhau.
6. Đánh giá FAR/FRR trên protocol được cố định trước để tránh data leakage.

## 10. Tài liệu tham khảo

- Dong et al., *WiFaKey: Generating Cryptographic Keys from Face in the Wild*, arXiv:2407.14804.
- ISO/IEC 24745:2022, Information technology — Security techniques — Biometric information protection.

## 11. Tóm tắt kết quả

Kết quả thực nghiệm hiện tại cho thấy Baseline Hard-BPSK hoạt động ổn định hơn các biến thể Soft-LLR đã thử nghiệm trong cấu hình nghiên cứu.

Soft-LLR chưa đạt được cải thiện rõ ràng trong các thí nghiệm hiện tại. Điều này không chứng minh Soft-LLR kém hiệu quả nói chung, mà cho thấy việc thay đổi phân phối LLR có thể cần đi kèm với calibration hoặc huấn luyện lại decoder.

Các phát hiện về kiểu dữ liệu, hash representation và cách xử lý masked bits cũng cho thấy việc kiểm soát chặt chẽ protocol, representation và test harness là rất quan trọng khi đánh giá hệ thống biometric cryptosystem.
