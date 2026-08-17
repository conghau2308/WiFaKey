use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Tạo một giá trị Gaussian từ HMAC(secret, row || col)
fn hmac_gaussian(secret: &[u8], row: usize, col: usize) -> f64 {
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC key");
    mac.update(&row.to_be_bytes());
    mac.update(&col.to_be_bytes());
    let digest = mac.finalize().into_bytes();
    
    let u = u32::from_be_bytes(digest[..4].try_into().unwrap()) as f64 / 0xFFFFFFFFu64 as f64;
    let v = u32::from_be_bytes(digest[4..8].try_into().unwrap()) as f64 / 0xFFFFFFFFu64 as f64;
    
    let r = (-2.0f64 * (u + 1e-15).ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * v;
    r * theta.cos()
}

pub fn generate_projection_matrix(user_secret: &[u8], dim: usize) -> Vec<f64> {
    let n = dim * dim;
    let mut m = vec![0.0f64; n];
    
    for i in 0..dim {
        for j in 0..dim {
            m[i * dim + j] = hmac_gaussian(user_secret, i, j);
        }
        // Chuẩn hoá hàng
        let start = i * dim;
        let mut norm_sq = 0.0;
        for j in 0..dim {
            norm_sq += m[start + j] * m[start + j];
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for j in 0..dim {
                m[start + j] /= norm;
            }
        }
    }
    m
}

pub fn biohash_project(embedding: &[f64], user_secret: &[u8]) -> Vec<f64> {
    let dim = embedding.len();
    let proj_matrix = generate_projection_matrix(user_secret, dim);
    let mut v_proj = vec![0.0f64; dim];
    for i in 0..dim {
        let mut sum = 0.0;
        for j in 0..dim {
            sum += proj_matrix[i * dim + j] * embedding[j];
        }
        v_proj[i] = sum;
    }
    let norm: f64 = v_proj.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        v_proj.iter_mut().for_each(|x| *x /= norm);
    }
    v_proj
}