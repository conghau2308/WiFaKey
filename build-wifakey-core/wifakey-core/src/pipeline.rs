//! Orchestration — nối toàn bộ chuỗi "frame raw -> embedding", port của
//! FaceProcessor.process() (Python). Đây là mảnh cuối cùng của pipeline ảnh.

use crate::embedding::EmbeddingExtractor;
use crate::face_alignment::{estimate_similarity_transform, warp_affine, REFERENCE_LANDMARKS};
use crate::face_detection::RetinaFaceDetector;
use crate::liveness::{crop_square_reflect, AntiSpoofSession};

/// Đúng 6 trạng thái đã chốt từ đầu (khớp FaceProcessor.process() Python):
/// no_image, no_face, low_confidence, spoof_detected, no_landmarks, success.
pub enum ProcessResult {
    NoImage,
    NoFace,
    LowConfidence,
    /// MỚI: khuôn mặt phát hiện được nhưng bbox chạm/gần mép khung hình —
    /// dấu hiệu khuôn mặt đang bị cắt ra ngoài camera (vd chỉ thấy nửa mặt).
    /// KHÔNG có trong taxonomy Python gốc — lỗ hổng phát hiện khi test
    /// camera thật, thêm mới để chặn trường hợp này.
    FaceOutOfFrame,
    SpoofDetected,
    /// LƯU Ý: trong thiết kế Rust hiện tại, `RetinaFaceDetector::detect_face`
    /// LUÔN trả kèm 5 keypoints khi tìm được mặt (không có đường nào thiếu
    /// landmarks riêng biệt như `getattr(face, "kps", None)` phía Python,
    /// vốn có thể trả None nếu model landmark phụ lỗi). Case này giữ lại để
    /// khớp taxonomy đã thống nhất, nhưng KHÔNG có nhánh code nào trong
    /// `process()` trả về nó — nếu sau này bạn thêm model landmark riêng có
    /// thể thất bại độc lập, đây là chỗ cần nối vào.
    NoLandmarks,
    Success { embedding: Vec<f32> },
}

fn bgr_to_rgb(bgr: &[u8], pixel_count: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; bgr.len()];
    for i in 0..pixel_count {
        rgb[i * 3] = bgr[i * 3 + 2];
        rgb[i * 3 + 1] = bgr[i * 3 + 1];
        rgb[i * 3 + 2] = bgr[i * 3];
    }
    rgb
}

pub struct FacePipeline {
    detector: RetinaFaceDetector,
    liveness: AntiSpoofSession,
    embedder: EmbeddingExtractor,
    confidence_threshold: f64,
    liveness_logit_threshold: f64,
    bbox_expansion_factor: f64,
}

impl FacePipeline {
    /// `liveness_threshold_prob`: xác suất (0-1, ví dụ 0.8) — sẽ tự chuyển
    /// sang logit threshold đúng công thức Python
    /// (`log(p / (1-p))`), không cần bạn tự tính tay.
    pub fn load(
        det_onnx_path: &str,
        liveness_onnx_path: &str,
        embedding_onnx_path: &str,
        confidence_threshold: f64,
        liveness_threshold_prob: f64,
        bbox_expansion_factor: f64,
    ) -> anyhow::Result<Self> {
        let detector = RetinaFaceDetector::load(det_onnx_path, 640)?;
        let liveness = AntiSpoofSession::load(liveness_onnx_path, 128)?;
        let embedder = EmbeddingExtractor::load(embedding_onnx_path)?;

        let p = liveness_threshold_prob.clamp(1e-6, 1.0 - 1e-6);
        let liveness_logit_threshold = (p / (1.0 - p)).ln();

        Ok(Self {
            detector,
            liveness,
            embedder,
            confidence_threshold,
            liveness_logit_threshold,
            bbox_expansion_factor,
        })
    }

    /// `frame_bgr`: ảnh phẳng HWC, BGR (khớp quy ước OpenCV/toàn bộ pipeline
    /// đã port) — nếu camera shell trả BGRA (Windows MediaCapture mặc định),
    /// PHẢI tự drop kênh alpha trước khi gọi hàm này (đã lưu ý từ câu trả
    /// lời rất sớm về BGRA -> BGR).
    pub fn process(
        &mut self,
        frame_bgr: &[u8],
        width: usize,
        height: usize,
    ) -> anyhow::Result<ProcessResult> {
        if frame_bgr.is_empty() || width == 0 || height == 0 {
            return Ok(ProcessResult::NoImage);
        }

        let detection =
            self.detector
                .detect_face(frame_bgr, width, height, self.confidence_threshold)?;
        let detection = match detection {
            None => return Ok(ProcessResult::NoFace),
            Some(d) => d,
        };

        if detection.confidence < self.confidence_threshold {
            return Ok(ProcessResult::LowConfidence);
        }

        // MỚI: kiểm tra bbox có chạm/gần mép khung hình không — dấu hiệu
        // khuôn mặt bị cắt ra ngoài (vd chỉ thấy nửa mặt). Margin 5% kích
        // thước khung hình mỗi bên — có thể chỉnh nếu quá chặt/lỏng khi
        // test thực tế.
        let margin_x = width as f64 * 0.05;
        let margin_y = height as f64 * 0.05;
        let [x1, y1, x2, y2] = detection.bbox;
        if x1 < margin_x || y1 < margin_y
            || x2 > width as f64 - margin_x
            || y2 > height as f64 - margin_y
        {
            return Ok(ProcessResult::FaceOutOfFrame);
        }

        // Liveness check trên crop vuông quanh bbox — đúng thứ tự Python
        // (kiểm liveness TRƯỚC khi align, không phải sau).
        let (crop_bgr, crop_size) = crop_square_reflect(
            frame_bgr,
            width,
            height,
            3,
            detection.bbox,
            self.bbox_expansion_factor,
        );
        let crop_rgb = bgr_to_rgb(&crop_bgr, crop_size * crop_size);

        let (is_live, _score) =
            self.liveness
                .check_liveness(&crop_rgb, crop_size, crop_size, self.liveness_logit_threshold)?;
        if !is_live {
            return Ok(ProcessResult::SpoofDetected);
        }

        // Align + embedding
        let m = estimate_similarity_transform(&detection.keypoints, &REFERENCE_LANDMARKS);
        let aligned_bgr = warp_affine(frame_bgr, width, height, 3, &m, 112, 112);
        let aligned_rgb = bgr_to_rgb(&aligned_bgr, 112 * 112);

        let embedding = self.embedder.get_embedding(&aligned_rgb)?;
        Ok(ProcessResult::Success { embedding })
    }
}