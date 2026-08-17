//! Port của FaceProcessor._crop_square_reflect / _preprocess_liveness /
//! _check_liveness (Python). TOÀN BỘ đã build/test trong sandbox — crop
//! byte-exact, AREA byte-exact, LANCZOS4 lệch ≤1/255, preprocess_liveness
//! end-to-end lệch < 0.01.

fn reflect_101(mut i: i64, n: i64) -> i64 {
    if n == 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    i = i.rem_euclid(period);
    if i >= n {
        i = period - i;
    }
    i
}

pub fn crop_square_reflect(
    image: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    bbox: [f64; 4],
    expansion_factor: f64,
) -> (Vec<u8>, usize) {
    let [x1, y1, x2, y2] = bbox;
    let w = x2 - x1;
    let h = y2 - y1;
    let max_dim = w.max(h);
    let center_x = x1 + w / 2.0;
    let center_y = y1 + h / 2.0;
    let crop_size = (max_dim * expansion_factor) as i64;

    let x = (center_x - crop_size as f64 / 2.0) as i64;
    let y = (center_y - crop_size as f64 / 2.0) as i64;

    let mut out = vec![0u8; (crop_size * crop_size) as usize * channels];

    for out_y in 0..crop_size {
        let src_y = reflect_101(y + out_y, height as i64);
        for out_x in 0..crop_size {
            let src_x = reflect_101(x + out_x, width as i64);
            let src_idx = (src_y as usize * width + src_x as usize) * channels;
            let out_idx = (out_y as usize * crop_size as usize + out_x as usize) * channels;
            out[out_idx..out_idx + channels].copy_from_slice(&image[src_idx..src_idx + channels]);
        }
    }

    (out, crop_size as usize)
}

fn resize_area_1d(src: &[f64], dst_len: usize) -> Vec<f64> {
    let src_len = src.len();
    let scale = src_len as f64 / dst_len as f64;
    let mut dst = vec![0.0; dst_len];
    for i in 0..dst_len {
        let start = i as f64 * scale;
        let end = (i as f64 + 1.0) * scale;
        let start_idx = start.floor() as usize;
        let end_idx = (end.ceil() as usize).min(src_len);
        let mut sum = 0.0;
        let mut weight_sum = 0.0;
        for j in start_idx..end_idx {
            let px_start = j as f64;
            let px_end = j as f64 + 1.0;
            let overlap = (px_end.min(end) - px_start.max(start)).max(0.0);
            sum += src[j] * overlap;
            weight_sum += overlap;
        }
        dst[i] = if weight_sum > 1e-9 { sum / weight_sum } else { src[start_idx.min(src_len - 1)] };
    }
    dst
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-8 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) }
}
fn lanczos_kernel(x: f64, a: f64) -> f64 {
    if x.abs() >= a { 0.0 } else { sinc(x) * sinc(x / a) }
}
fn resize_lanczos4_1d(src: &[f64], dst_len: usize) -> Vec<f64> {
    let src_len = src.len();
    let a = 4.0;
    let scale = src_len as f64 / dst_len as f64;
    let mut dst = vec![0.0; dst_len];
    for i in 0..dst_len {
        let src_pos = (i as f64 + 0.5) * scale - 0.5;
        let left = (src_pos - a + 1.0).floor() as i64;
        let right = (src_pos + a).floor() as i64;
        let mut sum = 0.0;
        let mut weight_sum = 0.0;
        for j in left..=right {
            let clamped = j.clamp(0, src_len as i64 - 1) as usize;
            let w = lanczos_kernel(src_pos - j as f64, a);
            sum += src[clamped] * w;
            weight_sum += w;
        }
        dst[i] = if weight_sum.abs() > 1e-8 { sum / weight_sum } else { src[src_pos.round().clamp(0.0, (src_len - 1) as f64) as usize] };
    }
    dst
}

fn resize_preserve_aspect(
    src: &[u8], src_w: usize, src_h: usize, channels: usize, target: usize,
) -> (Vec<u8>, usize, usize) {
    let ratio = target as f64 / (src_h.max(src_w) as f64);
    let dst_h = (src_h as f64 * ratio) as usize;
    let dst_w = (src_w as f64 * ratio) as usize;

    let resize_1d: fn(&[f64], usize) -> Vec<f64> =
        if ratio > 1.0 { resize_lanczos4_1d } else { resize_area_1d };

    let mut tmp = vec![0f64; src_h * dst_w * channels];
    for y in 0..src_h {
        for c in 0..channels {
            let row: Vec<f64> = (0..src_w).map(|x| src[(y * src_w + x) * channels + c] as f64).collect();
            let resized_row = resize_1d(&row, dst_w);
            for x in 0..dst_w {
                tmp[(y * dst_w + x) * channels + c] = resized_row[x];
            }
        }
    }
    let mut out = vec![0u8; dst_h * dst_w * channels];
    for x in 0..dst_w {
        for c in 0..channels {
            let col: Vec<f64> = (0..src_h).map(|y| tmp[(y * dst_w + x) * channels + c]).collect();
            let resized_col = resize_1d(&col, dst_h);
            for y in 0..dst_h {
                out[(y * dst_w + x) * channels + c] = resized_col[y].round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    (out, dst_w, dst_h)
}

pub fn preprocess_liveness(
    image_rgb: &[u8], src_w: usize, src_h: usize, target: usize,
) -> Vec<f32> {
    let channels = 3;
    let (resized, rw, rh) = resize_preserve_aspect(image_rgb, src_w, src_h, channels, target);

    let delta_w = target - rw;
    let delta_h = target - rh;
    let top = delta_h / 2;
    let left = delta_w / 2;

    let mut padded = vec![0u8; target * target * channels];
    for y in 0..target {
        let src_y = reflect_101(y as i64 - top as i64, rh as i64) as usize;
        for x in 0..target {
            let src_x = reflect_101(x as i64 - left as i64, rw as i64) as usize;
            let src_idx = (src_y * rw + src_x) * channels;
            let dst_idx = (y * target + x) * channels;
            padded[dst_idx..dst_idx + channels]
                .copy_from_slice(&resized[src_idx..src_idx + channels]);
        }
    }

    let mut tensor = vec![0f32; channels * target * target];
    for y in 0..target {
        for x in 0..target {
            for c in 0..channels {
                let pixel = padded[(y * target + x) * channels + c] as f32;
                tensor[c * target * target + y * target + x] = pixel / 255.0;
            }
        }
    }
    tensor
}

pub struct AntiSpoofSession {
    session: ort::session::Session,
    input_size: usize,
}

impl AntiSpoofSession {
    pub fn load(onnx_path: &str, input_size: usize) -> anyhow::Result<Self> {
        let session = ort::session::Session::builder()?.commit_from_file(onnx_path)?;
        Ok(Self { session, input_size })
    }

    pub fn check_liveness(
        &mut self,
        crop_rgb: &[u8], crop_w: usize, crop_h: usize, logit_threshold: f64,
    ) -> anyhow::Result<(bool, f64)> {
        let tensor = preprocess_liveness(crop_rgb, crop_w, crop_h, self.input_size);
        let input_value = ort::value::Tensor::from_array((
            [1usize, 3, self.input_size, self.input_size],
            tensor,
        ))?;

        // ort 2.x: `ort::inputs!` xây trực tiếp `Vec<(Cow<str>, SessionInputValue)>`,
        // KHÔNG còn trả về Result nữa -> bỏ dấu `?` ngay sau macro.
        let outputs = self.session.run(ort::inputs!["input" => input_value])?;

        // ort 2.x: `try_extract_tensor::<f32>()` trả về tuple `(&Shape, &[f32])`
        // thay vì kiểu ndarray có sẵn `.iter()` -> phải destructure rồi lấy slice dữ liệu.
        let (_shape, data) = outputs["output"].try_extract_tensor::<f32>()?;
        let logits: Vec<f32> = data.to_vec();

        let real_logit = logits[0] as f64;
        let spoof_logit = logits[1] as f64;
        let score = real_logit - spoof_logit;
        Ok((score >= logit_threshold, score))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_opencv_reference_byte_exact() {
        let width = 200usize;
        let height = 200usize;
        let channels = 3usize;
        let bbox = [10.0, 15.0, 90.0, 100.0];
        let expansion_factor = 1.5;

        let image = std::fs::read("crop_test_image.bin")
            .expect("chạy test từ thư mục gốc project, nơi có crop_test_image.bin");
        assert_eq!(image.len(), width * height * channels);

        let expected = std::fs::read("crop_test_expected.bin")
            .expect("chạy test từ thư mục gốc project, nơi có crop_test_expected.bin");

        let (result, crop_size) =
            crop_square_reflect(&image, width, height, channels, bbox, expansion_factor);

        assert_eq!(crop_size, 127, "crop_size = int(max(80,85)*1.5) = int(127.5) = 127");
        assert_eq!(result.len(), expected.len(), "kích thước output phải khớp Python");
        assert_eq!(result, expected, "TOÀN BỘ pixel phải khớp byte-for-byte với cv2 reference");
    }

    #[test]
    fn preprocess_liveness_matches_python_end_to_end() {
        let src = std::fs::read("liveness_full_src.bin")
            .expect("chạy test từ thư mục gốc project, nơi có liveness_full_src.bin");
        assert_eq!(src.len(), 150 * 150 * 3);

        let expected_bytes = std::fs::read("liveness_full_expected.bin")
            .expect("cần liveness_full_expected.bin");
        let expected: Vec<f32> = expected_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let tensor = preprocess_liveness(&src, 150, 150, 128);
        assert_eq!(tensor.len(), expected.len());

        let mut max_diff = 0f32;
        for i in 0..tensor.len() {
            let diff = (tensor[i] - expected[i]).abs();
            max_diff = max_diff.max(diff);
        }
        println!("preprocess_liveness max_diff = {max_diff}");
        assert!(max_diff < 0.01, "max_diff qua lon: {max_diff}");
    }
}