use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

/// Tạo hoán vị bằng cách gán mỗi vị trí một giá trị ngẫu nhiên từ HMAC, rồi sắp xếp
pub fn generate_permutation(salt_bytes: &[u8], n_bits: usize) -> Vec<usize> {
    let mut values: Vec<(usize, f64)> = Vec::with_capacity(n_bits);
    
    for i in 0..n_bits {
        let mut mac = HmacSha256::new_from_slice(salt_bytes).expect("HMAC key");
        mac.update(&i.to_be_bytes());
        let digest = mac.finalize().into_bytes();
        let val = u32::from_be_bytes(digest[..4].try_into().unwrap()) as f64 / 0xFFFFFFFFu64 as f64;
        values.push((i, val));
    }
    
    values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    values.into_iter().map(|(i, _)| i).collect()
}

pub fn apply_permutation<T: Clone>(data: &[T], perm: &[usize]) -> Vec<T> {
    perm.iter().map(|&i| data[i].clone()).collect()
}