# wifakey-core — Full pipeline scaffold

## Trạng thái build/test — đọc mục này trước tiên

| Phần | Đã build/test thật trong sandbox này? |
|---|---|
| `src/*.rs` (Rust core) | **CÓ** — `cargo build` + `cargo test` chạy thật, 15/15 test pass |
| `shell-ios-macos/*.swift` | KHÔNG — không có Xcode toolchain trong sandbox này, cần bạn tự build bằng Xcode |
| `shell-android/*.kt` | KHÔNG — không có Android SDK/Gradle trong sandbox này, cần bạn tự build bằng Android Studio |
| `shell-windows/*.cs`, `*.xaml` | KHÔNG — không có Windows/.NET toolchain trong sandbox này, cần bạn tự build bằng Visual Studio |

**Không có `shell-linux/`** — theo yêu cầu hiện tại chỉ dùng chữ ký thiết bị
dạng hardware, và Linux desktop không có đường hardware-backed signing đáng
tin cậy (đã bàn ở câu trả lời trước). Khi bạn nghiên cứu xong phương án
software-signing, thêm `shell-linux/` với `device_signature_software`
tương ứng — tách riêng khỏi `device_signature.rs` hiện tại (đọc comment đầu
file đó).

## Cấu trúc đầy đủ

```
wifakey-core/
├── Cargo.toml
├── wifakey.udl
├── python-reference/            # 1 file Python / 1 bước pipeline — KHÔNG gộp
│   ├── biohashing.py             # bước E4/V4 — CÓ implementation mẫu
│   ├── salted_permutation.py     # bước E5,7/V6,8 — TODO, để bạn dán code gốc
│   ├── binarization.py           # bước E6/V7 — TODO
│   └── ldpc_and_decode.py        # bước E12, V11-12 — TODO (đọc kỹ, phần khó nhất)
├── src/                          # core Rust — build/test thật
│   ├── lib.rs                    # entry point, wiring các module
│   ├── biohashing.rs              # XONG + test
│   ├── salted_permutation.rs      # XONG + test
│   ├── binarization.rs            # TODO — chờ threshold gốc của bạn
│   ├── margin_selection.rs        # XONG + test
│   ├── fuzzy_commitment.rs        # XONG + test
│   ├── ldpc.rs                    # TODO — xem comment, tách encode (port được)
│   │                               #        vs Neural-MS decode (export ONNX, không port tay)
│   ├── device_signature.rs        # XONG + test — chỉ dựng payload để ký, không tự ký
│   └── secure_memory.rs           # XONG (Unix) + test, TODO (Windows)
├── shell-ios-macos/              # dùng CHUNG SwiftUI cho iOS + macOS
│   └── WifakeyCapture/
│       ├── CameraCaptureView.swift    # UI hiện đại: oval guide, status pill, acrylic material
│       ├── CameraController.swift     # AVFoundation session, hook gọi core mỗi frame
│       └── DeviceSigning.swift        # Secure Enclave — CHỈ hardware, không fallback
├── shell-android/
│   └── app/src/main/java/com/wifakey/capture/
│       ├── CameraCaptureScreen.kt     # UI hiện đại: Compose + CameraX, cùng ngôn ngữ thiết kế
│       ├── CaptureViewModel.kt        # state machine, hook gọi core
│       └── DeviceSigning.kt           # StrongBox Keystore — CHỈ hardware, không fallback
└── shell-windows/
    ├── CaptureView.xaml               # WinUI3 markup, acrylic material
    ├── CaptureView.xaml.cs             # MediaCapture, state machine
    └── DeviceSigning.cs                # TPM qua CNG "Microsoft Platform Crypto Provider" — CHỈ hardware
```

## Trả lời câu hỏi: file Python gộp chung hay tách riêng?

**Tách riêng — 1 file / 1 bước pipeline**, mirror đúng cấu trúc `src/*.rs`
(vd `python-reference/binarization.py` ↔ `src/binarization.rs`). Lý do:

- Port từng bước độc lập, không phải kéo cả file khổng lồ ra đối chiếu mỗi lần.
- Dễ diff/review khi bạn cập nhật logic gốc sau này (vd tinh chỉnh lại
  threshold binarization mà không đụng gì tới fuzzy commitment).
- Đúng tinh thần "Bước 3" đã thống nhất ở lần trước: giữ file Python làm
  "spec sống" song song, không việc gì phải gộp lại.

Các module thuần thuật toán (margin_selection, fuzzy_commitment) tôi
implement thẳng bằng Rust, không cần file `.py` tương ứng vì logic quá đơn
giản để cần "bản gốc tham chiếu" — nhưng nếu bạn có sẵn code Python cho
phần này với quy ước khác, cứ đưa tôi, tôi chỉnh lại `.rs` cho khớp.

## Việc bạn cần làm tiếp — theo đúng thứ tự ưu tiên

1. **Dán code gốc vào 3 file `python-reference/*.py` còn TODO** (salted_permutation
   — dù tôi đã có bản Rust generic sẵn, đối chiếu xem có khớp thuật toán bạn
   dùng không; binarization — bắt buộc cần threshold gốc; ldpc_and_decode).
2. Port `binarization.rs` (dễ, chỉ còn thiếu threshold).
3. Port phần `encode()` trong `ldpc.rs` (thuần thuật toán, port được).
4. Export Neural-MS decoder sang ONNX, tích hợp qua crate `ort` vào
   `NeuralMsDecoder` trong `ldpc.rs` (đọc kỹ comment trong file, đây là phần
   duy nhất KHÔNG port tay).
5. Mở khoá `enroll()`/`verify()` thật trong `lib.rs` (đã có sẵn pseudocode
   comment chỉ đúng thứ tự gọi từng module).
6. Build thử từng shell — bắt đầu iOS/macOS trước (dùng chung code nhiều
   nhất), rồi Android, rồi Windows.

Mỗi khi bạn xong 1 bước, quay lại đây để tôi review/build thử (với phần
Rust — tôi build/test được thật trong sandbox này) trước khi bạn chuyển
sang bước kế.
