mod biohashing;
mod binarization;
mod device_signature;
mod embedding;
mod face_alignment;
mod face_detection;
mod liveness;
mod nms;
mod pipeline;
mod ffi;
mod fuzzy_commitment;
mod ldpc;
mod margin_selection;
mod salted_permutation;
mod secure_memory;
mod empirical_llr;

use anyhow::Result;

/// Đường dẫn model CLIENT cần — không còn neural_ms.onnx ở đây, vì decode
/// (Neural-MS) đã chuyển hẳn sang server theo quyết định mới nhất. Client
/// build (mobile/desktop) không cần đóng gói file .onnx này nữa, giảm size
/// app và không cần load ONNX Runtime cho bước verify.
pub struct ModelPaths<'a> {
    pub m_matrix: &'a str,
    pub generator_matrix_g: &'a str,
    pub empirical_llr_table: &'a str,
}

fn load_m_matrix(path: &str) -> Vec<f64> {
    let content = std::fs::read_to_string(path).expect("Không thể đọc M_matrix.txt");
    let mut m = Vec::new();
    for line in content.lines() {
        for num in line.split_whitespace() {
            m.push(num.parse::<f64>().unwrap());
        }
    }
    m
}

pub struct EnrollOutput {
    pub helper_data: Vec<bool>,
    pub reliability_mask: Vec<bool>,
    pub key_hash: Vec<u8>,
    pub service_salt: Vec<u8>,
}

pub fn enroll(
    embedding: &[f64],
    user_secret: &[u8],
    service_salt: &[u8],
    model_paths: &ModelPaths,
) -> Result<EnrollOutput> {
    let v_proj = biohashing::biohash_project(embedding, user_secret);
    let _ = secure_memory::lock_slice(&v_proj);

    let perm = salted_permutation::generate_permutation(service_salt, 1536);

    let m_matrix = load_m_matrix(model_paths.m_matrix);
    let intervals = vec![-0.029636395351665956, 0.0001636046483338635, 0.03006360464833386];
    let binarization::BinarizationResult { bits: b_full, margin } =
        binarization::binarize(&v_proj, &m_matrix, &intervals);

    let b_perm = salted_permutation::apply_permutation(&b_full, &perm);
    let margin_perm = salted_permutation::apply_permutation(&margin, &perm);

    let idx_sel = margin_selection::select_top_margin_indices(&margin_perm, 832);
    let b_sel: Vec<bool> = idx_sel.iter().map(|&i| b_perm[i] != 0).collect();
    let mask = margin_selection::build_mask(&idx_sel, 1536);

    let mut k = fuzzy_commitment::generate_random_key();
    let encoder = ldpc::LdpcEncoder::new(model_paths.generator_matrix_g)?;
    let c: Vec<bool> = encoder.encode(&k).iter().map(|&b| b != 0).collect();
    let helper_data = fuzzy_commitment::xor_bits(&b_sel, &c);
    let key_hash = fuzzy_commitment::key_hash(&k);

    // Xóa khóa khỏi bộ nhớ
    secure_memory::wipe(&mut k);

    Ok(EnrollOutput {
        helper_data,
        reliability_mask: mask,
        key_hash,
        service_salt: service_salt.to_vec(),
    })
}

/// Kết quả client tính xong, gửi lên server qua POST /verify/complete —
/// KHÔNG còn K'/success ở đây, vì client không còn decode nữa.
pub struct ClientVerifyPayload {
    pub llr: Vec<f64>,
}

/// Chạy TRÊN NATIVE APP. Dừng lại ở bước tính LLR (bước 11 trong bảng
/// Verify cũ) — KHÔNG gọi LDPC/Neural-MS nữa. Đây là toàn bộ phần "nhẹ"
/// (BioHashing, permutation, binarization, margin selection, empirical LLR)
/// — không cần ONNX Runtime, không cần file neural_ms.onnx trên máy user.
pub fn verify_client_prepare(
    embedding: &[f64],
    user_secret: &[u8],
    helper_data: &[bool],
    reliability_mask: &[bool],
    service_salt: &[u8],
    model_paths: &ModelPaths,
) -> Result<ClientVerifyPayload> {
    let v_proj = biohashing::biohash_project(embedding, user_secret);
    let _ = secure_memory::lock_slice(&v_proj);

    let perm = salted_permutation::generate_permutation(service_salt, 1536);

    let m_matrix = load_m_matrix(model_paths.m_matrix);
    let intervals = vec![-0.029636395351665956, 0.0001636046483338635, 0.03006360464833386];
    let binarization::BinarizationResult { bits: b_full, margin } =
        binarization::binarize(&v_proj, &m_matrix, &intervals);

    let b_perm = salted_permutation::apply_permutation(&b_full, &perm);
    let margin_perm = salted_permutation::apply_permutation(&margin, &perm);

    let idx_sel = margin_selection::indices_from_mask(reliability_mask);
    let b_sel: Vec<bool> = idx_sel.iter().map(|&i| b_perm[i] != 0).collect();
    let margin_sel: Vec<f64> = idx_sel.iter().map(|&i| margin_perm[i]).collect();

    let noisy = fuzzy_commitment::xor_bits(&b_sel, helper_data);

    let llr_table = empirical_llr::EmpiricalLlr::new(model_paths.empirical_llr_table)?;
    let llr: Vec<f64> = noisy.iter().enumerate()
        .map(|(i, &bit)| llr_table.compute_llr(bit, margin_sel[i]))
        .collect();

    Ok(ClientVerifyPayload { llr })
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    fn test_model_paths() -> ModelPaths<'static> {
        ModelPaths {
            m_matrix: "M_matrix.txt",
            generator_matrix_g: "generator_matrix_G.txt",
            empirical_llr_table: "empirical_llr_table.txt",
        }
    }

    #[test]
    fn verify_client_prepare_is_deterministic() {
        // Rust không còn tự khẳng định "verify thành công" được nữa (decode
        // giờ thuộc về FastAPI) — test này chỉ đảm bảo phần Rust CÒN LẠI
        // (đến hết bước tính LLR) tất định: cùng input phải luôn ra cùng
        // llr, để server/Python phía bên kia decode ra đúng cùng 1 kết quả
        // mỗi lần, không có gì ngẫu nhiên lẫn vào giữa đường.
        let embedding = vec![0.5; 512];
        let user_secret = b"test-secret-12345678";
        let service_salt = b"test-salt-87654321";
        let model_paths = test_model_paths();

        let enroll_out = enroll(&embedding, user_secret, service_salt, &model_paths)
            .expect("Enroll thất bại");

        let payload_a = verify_client_prepare(
            &embedding, user_secret, &enroll_out.helper_data,
            &enroll_out.reliability_mask, service_salt, &model_paths,
        ).expect("verify_client_prepare thất bại (lần 1)");

        let payload_b = verify_client_prepare(
            &embedding, user_secret, &enroll_out.helper_data,
            &enroll_out.reliability_mask, service_salt, &model_paths,
        ).expect("verify_client_prepare thất bại (lần 2)");

        assert_eq!(payload_a.llr, payload_b.llr, "Cùng input phải ra cùng LLR");
        assert_eq!(payload_a.llr.len(), 832, "LLR phải có đúng 832 phần tử (số bit đã chọn)");
    }

    #[test]
    fn different_embedding_gives_different_llr() {
        // Không khẳng định pass/fail (đó là việc của FastAPI sau khi decode)
        // — chỉ đảm bảo embedding khác nhau thực sự tạo ra LLR khác nhau,
        // tức là phần biohash/binarize/margin vẫn nhạy với input như kỳ vọng.
        let embedding1 = vec![0.5; 512];
        let embedding2 = vec![0.6; 512];
        let user_secret = b"test-secret-12345678";
        let service_salt = b"test-salt-87654321";
        let model_paths = test_model_paths();

        let enroll_out = enroll(&embedding1, user_secret, service_salt, &model_paths)
            .expect("Enroll thất bại");

        let payload1 = verify_client_prepare(
            &embedding1, user_secret, &enroll_out.helper_data,
            &enroll_out.reliability_mask, service_salt, &model_paths,
        ).expect("verify_client_prepare thất bại (embedding1)");

        let payload2 = verify_client_prepare(
            &embedding2, user_secret, &enroll_out.helper_data,
            &enroll_out.reliability_mask, service_salt, &model_paths,
        ).expect("verify_client_prepare thất bại (embedding2)");

        assert_ne!(payload1.llr, payload2.llr, "Embedding khác nhau phải cho LLR khác nhau");
    }

    // test_with_python_vectors (so khớp k_prime với decode gốc) đã CHUYỂN
    // SANG PHÍA PYTHON/FastAPI — vì decode không còn chạy trong Rust crate
    // này nữa. Viết test tương đương ở đó (pytest), so k_prime FastAPI decode
    // ra với original_k trong test_vectors.json, dùng đúng file JSON này.
}