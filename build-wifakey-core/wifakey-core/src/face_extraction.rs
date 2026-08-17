//! Face extraction pipeline: detection → alignment → AdaFace embedding.
//!
//! Hiện tại là STUB — bạn cần cung cấp:
//! - face_detection.onnx   (UltraFace / RetinaFace / CenterFace)
//! - adaface.onnx          (MobileFaceNet / AdaFace backbone)
//!
//! Khi có đủ file, thay thế các hàm stub bên dưới bằng logic thật.

use anyhow::{anyhow, Result};

/// Kết quả trích xuất khuôn mặt: embedding 512 chiều.
pub struct FaceEmbedding {
    pub vector: Vec<f64>,   // 512‑dim AdaFace embedding
}

/// Pipeline đầy đủ: ảnh thô → embedding.
///
/// * `image_bytes` - dữ liệu ảnh JPEG/PNG/BMP (từ camera hoặc file).
/// * `det_model`   - đường dẫn đến ONNX model face detection.
/// * `ada_model`   - đường dẫn đến ONNX model AdaFace.
pub fn extract_embedding(
    image_bytes: &[u8],
    det_model: &str,
    ada_model: &str,
) -> Result<FaceEmbedding> {
    // ── Bước 1: Face Detection ────────────────────────────────
    let face_bbox = detect_face(image_bytes, det_model)?;

    // ── Bước 2: Alignment (crop + affine warp) ────────────────
    let aligned = align_face(image_bytes, &face_bbox)?;

    // ── Bước 3: AdaFace embedding ─────────────────────────────
    let vector = adaface_embed(&aligned, ada_model)?;

    Ok(FaceEmbedding { vector })
}

// ── Stub: Face Detection ──────────────────────────────────────
fn detect_face(_image: &[u8], _model_path: &str) -> Result<FaceBbox> {
    // TODO: thay bằng ONNX inference với model UltraFace / RetinaFace.
    // Trả về bounding box (x1, y1, x2, y2) và 5 landmarks (nếu có).
    Err(anyhow!("Face detection not implemented yet"))
}

// ── Stub: Alignment ───────────────────────────────────────────
fn align_face(_image: &[u8], _bbox: &FaceBbox) -> Result<Vec<u8>> {
    // TODO: crop + affine transform để đưa khuôn mặt về kích thước 112×112.
    // Có thể dùng crate `image` để thao tác ảnh.
    Err(anyhow!("Face alignment not implemented yet"))
}

// ── Stub: AdaFace Embedding ───────────────────────────────────
fn adaface_embed(_aligned_rgb: &[u8], _model_path: &str) -> Result<Vec<f64>> {
    // TODO: nạp ONNX model AdaFace, chạy inference, lấy vector 512.
    Err(anyhow!("AdaFace embedding not implemented yet"))
}

// ── Cấu trúc dữ liệu ──────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct FaceBbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub landmarks: Option<[[f32; 2]; 5]>,  // 5 điểm (mắt, mũi, miệng)
}