// file: src/fuzzy_commitment.rs

use rand::Rng;
use sha2::{Sha256, Digest};

/// Tạo một khóa ngẫu nhiên 160-bit (20 bytes).
///
/// # Returns
/// Một vector chứa 20 byte ngẫu nhiên.
pub fn generate_random_key() -> Vec<u8> {
    let mut rng = rand::thread_rng();
    let mut key = vec![0u8; 20];
    rng.fill(&mut key[..]);
    key
}

/// Tính SHA256 của một slice byte.
///
/// # Arguments
/// * `k` - Khóa đầu vào (dạng byte).
///
/// # Returns
/// Một vector chứa 32 byte hash.
pub fn key_hash(k: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(k);
    hasher.finalize().to_vec()
}

/// Thực hiện phép XOR giữa hai vector bit (biểu diễn dưới dạng `bool`).
///
/// # Arguments
/// * `a`, `b` - Hai slice `bool` có cùng độ dài.
///
/// # Returns
/// Một vector `bool` mới chứa kết quả phép XOR.
pub fn xor_bits(a: &[bool], b: &[bool]) -> Vec<bool> {
    assert_eq!(a.len(), b.len(), "a and b must have the same length");
    a.iter().zip(b.iter()).map(|(&x, &y)| x ^ y).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_random_key_length() {
        let key = generate_random_key();
        assert_eq!(key.len(), 20);
    }

    #[test]
    fn test_key_hash_length() {
        let key = b"test_key_123456789";
        let hash = key_hash(key);
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_xor_bits() {
        let a = vec![true, false, true];
        let b = vec![false, false, true];
        let result = xor_bits(&a, &b);
        assert_eq!(result, vec![true, false, false]);
    }
}
