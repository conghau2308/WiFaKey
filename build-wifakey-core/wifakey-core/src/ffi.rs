//! Lớp FFI mỏng CHỈ dùng cho Windows/C# — iOS/Android dùng UniFFI (được hỗ
//! trợ chính thức), không đi qua file này. Đây là phương án 2 đã chốt: tự
//! viết C-ABI thay vì phụ thuộc `uniffi-bindgen-cs` (thư viện cộng đồng,
//! không chính thức).
//!
//! Giao thức: C# gửi 1 chuỗi JSON qua con trỏ C string, nhận lại 1 chuỗi
//! JSON khác (kết quả hoặc lỗi). Đơn giản nhất để không phải tự viết struct
//! marshalling nhị phân qua C ABI — enroll/verify không phải hot loop
//! (gọi vài lần mỗi phiên, không phải hàng nghìn lần/giây), nên overhead
//! JSON không đáng lo.

use crate::{enroll, verify_client_prepare, ModelPaths};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[derive(Deserialize)]
struct EnrollRequest {
    embedding: Vec<f64>,
    user_secret: Vec<u8>,
    service_salt: Vec<u8>,
    m_matrix_path: String,
    generator_matrix_g_path: String,
    empirical_llr_table_path: String,
}

#[derive(Serialize)]
struct EnrollResponse {
    helper_data: Vec<bool>,
    reliability_mask: Vec<bool>,
    key_hash: Vec<u8>,
    service_salt: Vec<u8>,
}

#[derive(Deserialize)]
struct VerifyPrepareRequest {
    embedding: Vec<f64>,
    user_secret: Vec<u8>,
    helper_data: Vec<bool>,
    reliability_mask: Vec<bool>,
    service_salt: Vec<u8>,
    m_matrix_path: String,
    generator_matrix_g_path: String,
    empirical_llr_table_path: String,
}

#[derive(Serialize)]
struct VerifyPrepareResponse {
    llr: Vec<f64>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn ok_json<T: Serialize>(value: &T) -> String {
    // Nếu chính bước serialize JSON kết quả cũng lỗi (hiếm, nhưng về mặt lý
    // thuyết có thể xảy ra), vẫn phải trả về 1 JSON hợp lệ cho C# parse
    // được, không panic qua FFI boundary (panic xuyên qua extern "C" là
    // undefined behavior).
    serde_json::to_string(value)
        .unwrap_or_else(|e| format!("{{\"error\":\"serialize thất bại: {e}\"}}"))
}

fn err_json(msg: impl std::fmt::Display) -> String {
    serde_json::to_string(&ErrorResponse { error: msg.to_string() })
        .unwrap_or_else(|_| "{\"error\":\"không rõ lỗi\"}".to_string())
}

fn to_c_string(json: String) -> *mut c_char {
    match CString::new(json) {
        Ok(s) => s.into_raw(),
        Err(_) => CString::new("{\"error\":\"json chứa null byte\"}")
            .expect("chuỗi lỗi cố định không thể chứa null byte")
            .into_raw(),
    }
}

/// Bọc phần thân của MỖI hàm FFI bằng `catch_unwind` — đây KHÔNG phải tối
/// ưu tuỳ chọn mà là yêu cầu bắt buộc cho mọi hàm `extern "C"`. Lý do:
/// panic (từ `.expect()`/`.unwrap()` ở bất kỳ đâu trong lib.rs/binarization.rs/
/// ldpc.rs...) mà xuyên qua ranh giới `extern "C"` sẽ làm Rust ABORT TOÀN BỘ
/// TIẾN TRÌNH gọi vào (đã tái hiện thật: thiếu M_matrix.txt -> app C# crash
/// hoàn toàn, không có cách nào try/catch từ phía C#). `catch_unwind` chặn
/// panic lại NGAY TRONG hàm Rust, trước khi nó kịp lan ra ranh giới FFI,
/// rồi chuyển thành 1 JSON lỗi bình thường — an toàn cho MỌI panic tương
/// lai, không chỉ riêng lỗi thiếu file lần này.
fn run_ffi_guarded<F>(f: F) -> String
where
    F: FnOnce() -> anyhow::Result<String> + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(Ok(json)) => json,
        Ok(Err(e)) => err_json(e),
        Err(panic_payload) => {
            let msg = panic_payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic_payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic không rõ nội dung".to_string());
            err_json(format!("lỗi nội bộ (panic đã được chặn, không crash app): {msg}"))
        }
    }
}

/// # Safety
/// `input_json` phải là con trỏ hợp lệ tới chuỗi C UTF-8 kết thúc bằng NUL
/// do C# cấp (Marshal.StringToHGlobalAnsi hoặc P/Invoke tự động marshal
/// `string` -> `char*`). Con trỏ trả về PHẢI được giải phóng bằng
/// `wifakey_free_string` — không dùng `Marshal.FreeHGlobal` hay bất kỳ cách
/// nào khác phía C#, vì bộ nhớ được Rust allocator cấp phát, phải do đúng
/// Rust allocator giải phóng.
#[no_mangle]
pub unsafe extern "C" fn wifakey_enroll(input_json: *const c_char) -> *mut c_char {
    let json = run_ffi_guarded(move || {
        let input_str = CStr::from_ptr(input_json).to_str()?;
        let req: EnrollRequest = serde_json::from_str(input_str)?;
        let model_paths = ModelPaths {
            m_matrix: &req.m_matrix_path,
            generator_matrix_g: &req.generator_matrix_g_path,
            empirical_llr_table: &req.empirical_llr_table_path,
        };
        let output = enroll(&req.embedding, &req.user_secret, &req.service_salt, &model_paths)?;
        Ok(ok_json(&EnrollResponse {
            helper_data: output.helper_data,
            reliability_mask: output.reliability_mask,
            key_hash: output.key_hash,
            service_salt: output.service_salt,
        }))
    });
    to_c_string(json)
}

/// # Safety
/// Xem ghi chú `wifakey_enroll` — cùng quy tắc con trỏ vào/ra.
#[no_mangle]
pub unsafe extern "C" fn wifakey_verify_prepare(input_json: *const c_char) -> *mut c_char {
    let json = run_ffi_guarded(move || {
        let input_str = CStr::from_ptr(input_json).to_str()?;
        let req: VerifyPrepareRequest = serde_json::from_str(input_str)?;
        let model_paths = ModelPaths {
            m_matrix: &req.m_matrix_path,
            generator_matrix_g: &req.generator_matrix_g_path,
            empirical_llr_table: &req.empirical_llr_table_path,
        };
        let payload = verify_client_prepare(
            &req.embedding,
            &req.user_secret,
            &req.helper_data,
            &req.reliability_mask,
            &req.service_salt,
            &model_paths,
        )?;
        Ok(ok_json(&VerifyPrepareResponse { llr: payload.llr }))
    });
    to_c_string(json)
}

/// C# BẮT BUỘC gọi hàm này sau khi đọc xong chuỗi trả về từ
/// `wifakey_enroll`/`wifakey_verify_prepare` — nếu không sẽ leak memory,
/// vì Rust không tự biết khi nào phía C# đã đọc xong để giải phóng.
///
/// # Safety
/// `s` phải là con trỏ do chính `wifakey_enroll`/`wifakey_verify_prepare`
/// trả về, gọi đúng 1 lần (double-free là undefined behavior).
#[no_mangle]
pub unsafe extern "C" fn wifakey_free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}


// ============================================================================
// Pipeline (camera -> embedding) — khác 2 hàm trên ở 2 điểm:
//   1. Giữ session sống qua NHIỀU lần gọi (load 1 lần, gọi process_frame mỗi
//      frame camera, không load lại model mỗi lần — đúng yêu cầu tài nguyên
//      đã bàn từ đầu cho auto-capture).
//   2. Frame ảnh qua CON TRỎ THÔ, không qua JSON — ảnh vài MB/frame ở
//      15-30fps sẽ lãng phí CPU nếu encode/decode JSON mỗi lần.
// ============================================================================

use crate::pipeline::{FacePipeline, ProcessResult};

#[derive(Deserialize)]
struct PipelineLoadRequest {
    det_onnx_path: String,
    liveness_onnx_path: String,
    embedding_onnx_path: String,
    confidence_threshold: f64,
    liveness_threshold_prob: f64,
    bbox_expansion_factor: f64,
}

#[derive(Serialize)]
struct PipelineLoadResponse {
    /// Con trỏ tới FacePipeline đã Box, ép kiểu sang u64 để qua được FFI
    /// boundary an toàn — C# giữ số này, KHÔNG được tự ý diễn giải/tính
    /// toán trên nó, chỉ truyền nguyên vẹn vào process_frame/free.
    handle: u64,
}

/// # Safety
/// Xem ghi chú `wifakey_enroll` — cùng quy tắc con trỏ vào/ra cho
/// `input_json`. Trả về `handle` — PHẢI gọi `wifakey_pipeline_free` đúng 1
/// lần khi không dùng nữa, nếu không sẽ leak toàn bộ 3 model đã load (rất
/// nặng, không giống leak 1 chuỗi string nhỏ).
#[no_mangle]
pub unsafe extern "C" fn wifakey_pipeline_load(input_json: *const c_char) -> *mut c_char {
    let json = run_ffi_guarded(move || {
        let input_str = CStr::from_ptr(input_json).to_str()?;
        let req: PipelineLoadRequest = serde_json::from_str(input_str)?;
        let pipeline = FacePipeline::load(
            &req.det_onnx_path,
            &req.liveness_onnx_path,
            &req.embedding_onnx_path,
            req.confidence_threshold,
            req.liveness_threshold_prob,
            req.bbox_expansion_factor,
        )?;
        let handle = Box::into_raw(Box::new(pipeline)) as u64;
        Ok(ok_json(&PipelineLoadResponse { handle }))
    });
    to_c_string(json)
}

/// JSON trả về dùng "status" làm tag — khớp ĐÚNG 6 trạng thái đã thống nhất
/// cho CaptureState phía UI: "no_image"|"no_face"|"low_confidence"|
/// "spoof_detected"|"no_landmarks"|"success" (kèm "embedding" nếu success).
#[derive(Serialize)]
#[serde(tag = "status")]
enum ProcessFrameResponse {
    #[serde(rename = "no_image")]
    NoImage,
    #[serde(rename = "no_face")]
    NoFace,
    #[serde(rename = "low_confidence")]
    LowConfidence,
    #[serde(rename = "face_out_of_frame")]
    FaceOutOfFrame,
    #[serde(rename = "spoof_detected")]
    SpoofDetected,
    #[serde(rename = "no_landmarks")]
    NoLandmarks,
    #[serde(rename = "success")]
    Success { embedding: Vec<f32> },
}

/// # Safety
/// `handle` phải là giá trị do chính `wifakey_pipeline_load` trả về, chưa
/// từng được truyền vào `wifakey_pipeline_free`. `frame_bgr` phải trỏ tới
/// đúng `frame_len` byte hợp lệ, bố cục HWC/BGR (3 kênh, không alpha — nếu
/// camera trả BGRA, C# phải tự drop kênh alpha TRƯỚC khi gọi hàm này).
#[no_mangle]
pub unsafe extern "C" fn wifakey_pipeline_process_frame(
    handle: u64,
    frame_bgr: *const u8,
    frame_len: usize,
    width: u32,
    height: u32,
) -> *mut c_char {
    let json = run_ffi_guarded(move || {
        if handle == 0 {
            anyhow::bail!("handle rỗng (0) — pipeline chưa load hoặc đã free");
        }
        let pipeline = &mut *(handle as *mut FacePipeline);
        let frame_slice = std::slice::from_raw_parts(frame_bgr, frame_len);
        let result = pipeline.process(frame_slice, width as usize, height as usize)?;

        let response = match result {
            ProcessResult::NoImage => ProcessFrameResponse::NoImage,
            ProcessResult::NoFace => ProcessFrameResponse::NoFace,
            ProcessResult::LowConfidence => ProcessFrameResponse::LowConfidence,
            ProcessResult::FaceOutOfFrame => ProcessFrameResponse::FaceOutOfFrame,
            ProcessResult::SpoofDetected => ProcessFrameResponse::SpoofDetected,
            ProcessResult::NoLandmarks => ProcessFrameResponse::NoLandmarks,
            ProcessResult::Success { embedding } => ProcessFrameResponse::Success { embedding },
        };
        Ok(ok_json(&response))
    });
    to_c_string(json)
}

/// C# BẮT BUỘC gọi đúng 1 lần khi không dùng pipeline nữa (vd đóng màn hình
/// capture) — giải phóng cả 3 model đã load.
///
/// # Safety
/// `handle` phải do `wifakey_pipeline_load` trả về, và đây phải là lần gọi
/// `free` DUY NHẤT cho handle này (double-free là undefined behavior).
#[no_mangle]
pub unsafe extern "C" fn wifakey_pipeline_free(handle: u64) {
    if handle != 0 {
        let _ = Box::from_raw(handle as *mut FacePipeline);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enroll_roundtrip_via_json() {
        let request = serde_json::json!({
            "embedding": vec![0.5f64; 512],
            "user_secret": b"test-secret-12345678".to_vec(),
            "service_salt": b"test-salt-87654321".to_vec(),
            "m_matrix_path": "M_matrix.txt",
            "generator_matrix_g_path": "generator_matrix_G.txt",
            "empirical_llr_table_path": "empirical_llr_table.txt",
        });
        let input = CString::new(request.to_string()).unwrap();

        let result_ptr = unsafe { wifakey_enroll(input.as_ptr()) };
        let result_str = unsafe { CStr::from_ptr(result_ptr) }.to_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(result_str).unwrap();

        assert!(parsed.get("error").is_none(), "không được có lỗi: {result_str}");
        assert!(parsed.get("helper_data").is_some());
        assert!(parsed.get("key_hash").is_some());

        unsafe { wifakey_free_string(result_ptr) };
    }

    #[test]
    fn malformed_json_returns_error_not_panic() {
        let input = CString::new("{ khong phai json hop le }").unwrap();
        let result_ptr = unsafe { wifakey_enroll(input.as_ptr()) };
        let result_str = unsafe { CStr::from_ptr(result_ptr) }.to_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(result_str).unwrap();

        assert!(parsed.get("error").is_some(), "JSON sai định dạng phải trả lỗi, không panic");

        unsafe { wifakey_free_string(result_ptr) };
    }
}