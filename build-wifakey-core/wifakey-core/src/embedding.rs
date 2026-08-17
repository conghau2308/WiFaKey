//! Port của AdaFaceExtractor._preprocess_image() + get_feature_vector()
//! (Python). EmbeddingExtractor::get_embedding() ĐÃ BIÊN DỊCH THÀNH CÔNG
//! với ort 2.0.0-rc.13 thật trong sandbox — chưa chạy được (thiếu
//! onnxruntime.dll + file .onnx), bạn cần tự chạy test tích hợp cuối file
//! trên Windows để xác nhận runtime.

pub fn preprocess(aligned_rgb_112: &[u8]) -> Vec<f32> {
    assert_eq!(aligned_rgb_112.len(), 112 * 112 * 3, "input phải đúng 112x112x3 RGB");

    let mut tensor = vec![0f32; 3 * 112 * 112];
    for y in 0..112 {
        for x in 0..112 {
            for c in 0..3 {
                let pixel = aligned_rgb_112[(y * 112 + x) * 3 + c] as f32;
                let normalized = (pixel / 255.0 - 0.5) / 0.5;
                tensor[c * 112 * 112 + y * 112 + x] = normalized;
            }
        }
    }
    tensor
}

pub fn l2_normalize(embedding: &mut [f32]) {
    let norm: f32 = embedding.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in embedding.iter_mut() {
            *v /= norm;
        }
    }
}

pub struct EmbeddingExtractor {
    session: ort::session::Session,
}

impl EmbeddingExtractor {
    pub fn load(onnx_path: &str) -> anyhow::Result<Self> {
        let session = ort::session::Session::builder()?.commit_from_file(onnx_path)?;
        Ok(Self { session })
    }

    // rc.13: Session::run yêu cầu &mut self
    pub fn get_embedding(&mut self, aligned_rgb_112: &[u8]) -> anyhow::Result<Vec<f32>> {
        let tensor = preprocess(aligned_rgb_112);
        let input_value = ort::value::Tensor::from_array(([1usize, 3, 112, 112], tensor))?;

        // rc.13: inputs! không còn trả về Result -> bỏ dấu ? sau nó
        let outputs = self.session.run(ort::inputs!["input" => input_value])?;

        // rc.13: try_extract_tensor trả về tuple (&Shape, &[f32]) chứ không phải ArrayView
        let (_shape, data) = outputs["embedding"].try_extract_tensor::<f32>()?;
        let mut embedding: Vec<f32> = data.to_vec();

        l2_normalize(&mut embedding);
        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_matches_python_reference() {
        let image = std::fs::read("aligned_face_112.bin")
            .expect("chạy test từ thư mục gốc project");
        assert_eq!(image.len(), 112 * 112 * 3);

        let expected_bytes = std::fs::read("expected_preprocessed_tensor.bin")
            .expect("cần expected_preprocessed_tensor.bin");
        let expected: Vec<f32> = expected_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        let tensor = preprocess(&image);
        assert_eq!(tensor.len(), expected.len());

        for i in 0..tensor.len() {
            let diff = (tensor[i] - expected[i]).abs();
            assert!(diff < 1e-6, "tensor[{i}] khác expected (diff {diff})");
        }
    }

    #[test]
    fn l2_normalize_produces_unit_norm() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    // #[ignore] // bỏ #[ignore] khi có adaface_ir101.onnx + onnxruntime.dll thật
    fn full_inference_matches_expected_embedding() {
        let image = std::fs::read("aligned_face_112.bin").unwrap();
        let mut extractor = EmbeddingExtractor::load("adaface_ir101.onnx").unwrap();
        let embedding = extractor.get_embedding(&image).unwrap();

        let expected_bytes = std::fs::read("expected_embedding.bin").unwrap();
        let expected: Vec<f32> = expected_bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        assert_eq!(embedding.len(), 512);
        for i in 0..512 {
            assert!((embedding[i] - expected[i]).abs() < 1e-4, "embedding[{i}] lệch quá nhiều");
        }
    }
}