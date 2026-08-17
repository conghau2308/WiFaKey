//! Port của InsightFaceDetector.detect() (phần post-processing — KHÔNG bao
//! gồm chạy ONNX inference). LƯU Ý: đây là port của bản `InsightFaceDetector`
//! (Python tự viết lại), KHÔNG PHẢI `insightface.app.FaceAnalysis` chính
//! thức — 2 pipeline lệch nhau nhẹ (~0.1-0.3 pixel, ~0.005 score), đã đo
//! thực nghiệm, chấp nhận như giới hạn đã biết.

use crate::nms::nms;

const STRIDES: [usize; 3] = [8, 16, 32];
const NUM_ANCHORS: usize = 2;
const NMS_THRESH: f64 = 0.4;

pub struct DetectionResult {
    pub bbox: [f64; 4],
    pub keypoints: [[f64; 2]; 5],
    pub confidence: f64,
}

fn generate_anchors(det_h: usize, det_w: usize, stride: usize) -> Vec<[f64; 2]> {
    let fh = det_h / stride;
    let fw = det_w / stride;
    let mut centers = Vec::with_capacity(fh * fw * NUM_ANCHORS);
    for gy in 0..fh {
        for gx in 0..fw {
            let cx = (gx * stride) as f64;
            let cy = (gy * stride) as f64;
            for _ in 0..NUM_ANCHORS {
                centers.push([cx, cy]);
            }
        }
    }
    centers
}

fn dist2bbox(anchor: [f64; 2], delta: &[f32]) -> [f64; 4] {
    [
        anchor[0] - delta[0] as f64,
        anchor[1] - delta[1] as f64,
        anchor[0] + delta[2] as f64,
        anchor[1] + delta[3] as f64,
    ]
}

fn dist2kps(anchor: [f64; 2], delta: &[f32]) -> [[f64; 2]; 5] {
    let mut kps = [[0.0; 2]; 5];
    for k in 0..5 {
        kps[k][0] = anchor[0] + delta[2 * k] as f64;
        kps[k][1] = anchor[1] + delta[2 * k + 1] as f64;
    }
    kps
}

pub fn detect(
    outputs: &[&[f32]; 9],
    det_size: (usize, usize),
    scale: f64,
    confidence_threshold: f64,
) -> Option<DetectionResult> {
    let (det_w, det_h) = det_size;
    let fmc = 3;

    let mut all_scores: Vec<f64> = Vec::new();
    let mut all_bboxes: Vec<[f64; 4]> = Vec::new();
    let mut all_kpss: Vec<[[f64; 2]; 5]> = Vec::new();

    for (i, &stride) in STRIDES.iter().enumerate() {
        let scores = outputs[i];
        let bbox_deltas = outputs[i + fmc];
        let kps_deltas = outputs[i + fmc * 2];
        let anchors = generate_anchors(det_h, det_w, stride);

        for (idx, &score) in scores.iter().enumerate() {
            let score = score as f64;
            if score < confidence_threshold {
                continue;
            }
            let anchor = anchors[idx];

            let bd = &bbox_deltas[idx * 4..idx * 4 + 4];
            let scaled_bd: Vec<f32> = bd.iter().map(|&v| v * stride as f32).collect();
            let bbox = dist2bbox(anchor, &scaled_bd);

            let kd = &kps_deltas[idx * 10..idx * 10 + 10];
            let scaled_kd: Vec<f32> = kd.iter().map(|&v| v * stride as f32).collect();
            let kps = dist2kps(anchor, &scaled_kd);

            all_scores.push(score);
            all_bboxes.push(bbox);
            all_kpss.push(kps);
        }
    }

    if all_scores.is_empty() {
        return None;
    }

    let keep = nms(&all_bboxes, &all_scores, NMS_THRESH);
    if keep.is_empty() {
        return None;
    }

    let mut best_idx = keep[0];
    let mut best_area = f64::MIN;
    for &i in &keep {
        let b = all_bboxes[i];
        let area = ((b[2] - b[0]) / scale) * ((b[3] - b[1]) / scale);
        if area > best_area {
            best_area = area;
            best_idx = i;
        }
    }

    let b = all_bboxes[best_idx];
    let bbox = [b[0] / scale, b[1] / scale, b[2] / scale, b[3] / scale];
    let mut kps = all_kpss[best_idx];
    for p in kps.iter_mut() {
        p[0] /= scale;
        p[1] /= scale;
    }

    Some(DetectionResult {
        bbox,
        keypoints: kps,
        confidence: all_scores[best_idx],
    })
}

/// Resize-giữ-tỷ-lệ + pad về đúng det_size (mặc định 640x640), trả về
/// (blob NCHW đã normalize, scale đã dùng) — khớp đúng logic trong
/// InsightFaceDetector.detect() (Python): scale = min(dw/iw, dh/ih), resize
/// theo scale đó, pad phần còn thiếu bằng 0 ở góc dưới-phải.
fn preprocess_frame(
    frame_bgr: &[u8],
    frame_w: usize,
    frame_h: usize,
    det_size: usize,
) -> (Vec<f32>, f64) {
    let scale = (det_size as f64 / frame_w as f64).min(det_size as f64 / frame_h as f64);
    let nw = (frame_w as f64 * scale) as usize;
    let nh = (frame_h as f64 * scale) as usize;

    let mut canvas = vec![0u8; det_size * det_size * 3];
    for y in 0..nh {
        let src_y = (y as f64 + 0.5) / scale - 0.5;
        let y0 = src_y.floor().clamp(0.0, (frame_h - 1) as f64) as usize;
        let y1 = (y0 + 1).min(frame_h - 1);
        let wy = (src_y - y0 as f64).clamp(0.0, 1.0);

        for x in 0..nw {
            let src_x = (x as f64 + 0.5) / scale - 0.5;
            let x0 = src_x.floor().clamp(0.0, (frame_w - 1) as f64) as usize;
            let x1 = (x0 + 1).min(frame_w - 1);
            let wx = (src_x - x0 as f64).clamp(0.0, 1.0);

            for c in 0..3 {
                let p00 = frame_bgr[(y0 * frame_w + x0) * 3 + c] as f64;
                let p01 = frame_bgr[(y0 * frame_w + x1) * 3 + c] as f64;
                let p10 = frame_bgr[(y1 * frame_w + x0) * 3 + c] as f64;
                let p11 = frame_bgr[(y1 * frame_w + x1) * 3 + c] as f64;
                let top = p00 * (1.0 - wx) + p01 * wx;
                let bottom = p10 * (1.0 - wx) + p11 * wx;
                let value = (top * (1.0 - wy) + bottom * wy).round() as u8;
                canvas[(y * det_size + x) * 3 + c] = value;
            }
        }
    }

    let mut blob = vec![0f32; 3 * det_size * det_size];
    for y in 0..det_size {
        for x in 0..det_size {
            for c in 0..3 {
                let pixel = canvas[(y * det_size + x) * 3 + c] as f32;
                let normalized = (pixel - 127.5) / 128.0;
                blob[c * det_size * det_size + y * det_size + x] = normalized;
            }
        }
    }

    (blob, scale)
}

pub struct RetinaFaceDetector {
    session: ort::session::Session,
    det_size: usize,
}

impl RetinaFaceDetector {
    pub fn load(onnx_path: &str, det_size: usize) -> anyhow::Result<Self> {
        let session = ort::session::Session::builder()?.commit_from_file(onnx_path)?;
        Ok(Self { session, det_size })
    }

    pub fn detect_face(
        &mut self,
        frame_bgr: &[u8],
        frame_w: usize,
        frame_h: usize,
        confidence_threshold: f64,
    ) -> anyhow::Result<Option<DetectionResult>> {
        let (blob, scale) = preprocess_frame(frame_bgr, frame_w, frame_h, self.det_size);
        let input_value =
            ort::value::Tensor::from_array(([1usize, 3, self.det_size, self.det_size], blob))?;

        // ort 2.x: `ort::inputs!` xây trực tiếp Vec<(Cow<str>, SessionInputValue)>,
        // KHÔNG còn trả về Result -> bỏ dấu `?` ngay sau macro.
        // `session.run()` cũng cần `&mut self` trong bản ort đang dùng.
        let outputs = self.session.run(ort::inputs!["input.1" => input_value])?;

        // Tên output đã xác nhận từ introspect det_10g.onnx thật (không đoán).
        let names = ["448", "471", "494", "451", "474", "497", "454", "477", "500"];
        let arrays: Vec<Vec<f32>> = names
            .iter()
            .map(|&name| -> anyhow::Result<Vec<f32>> {
                // ort 2.x: try_extract_tensor::<f32>() trả về tuple (&Shape, &[f32])
                // thay vì kiểu có sẵn .iter() -> destructure rồi lấy slice dữ liệu.
                let (_shape, data) = outputs[name].try_extract_tensor::<f32>()?;
                Ok(data.to_vec())
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let refs: [&[f32]; 9] = [
            &arrays[0], &arrays[1], &arrays[2], &arrays[3], &arrays[4],
            &arrays[5], &arrays[6], &arrays[7], &arrays[8],
        ];

        Ok(detect(&refs, (self.det_size, self.det_size), scale, confidence_threshold))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_f32_bin(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|e| panic!("không đọc được {path}: {e}"));
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    #[test]
    fn matches_python_reimplementation_on_real_tensors() {
        let tensors: Vec<Vec<f32>> = (0..9).map(|i| load_f32_bin(&format!("det10g_out{i}.bin"))).collect();
        let refs: [&[f32]; 9] = [
            &tensors[0], &tensors[1], &tensors[2], &tensors[3], &tensors[4],
            &tensors[5], &tensors[6], &tensors[7], &tensors[8],
        ];

        let result = detect(&refs, (640, 640), 2.56, 0.5)
            .expect("phải detect được khuôn mặt trong ảnh test");

        let expected_bbox = [65.450, 58.164, 170.064, 199.170];
        let expected_score = 0.8279;

        for i in 0..4 {
            let diff = (result.bbox[i] - expected_bbox[i]).abs();
            assert!(diff < 0.01, "bbox[{i}] = {} khác expected {} (diff {diff})", result.bbox[i], expected_bbox[i]);
        }
        assert!((result.confidence - expected_score).abs() < 0.001);

        let expected_kps = [
            [99.263, 113.700], [144.472, 113.130], [125.676, 136.073],
            [100.612, 157.835], [144.650, 157.156],
        ];
        for k in 0..5 {
            for c in 0..2 {
                let diff = (result.keypoints[k][c] - expected_kps[k][c]).abs();
                assert!(diff < 0.01, "kps[{k}][{c}] khác expected");
            }
        }
    }
}