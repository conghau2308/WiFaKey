use std::fs;

/// LDPC Encoder: dùng ma trận sinh G (160 x 832) trên GF(2).
pub struct LdpcEncoder {
    g: Vec<Vec<u8>>,
}

impl LdpcEncoder {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let g: Vec<Vec<u8>> = content
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|s| s.parse::<u8>().unwrap())
                    .collect()
            })
            .collect();
        anyhow::ensure!(g.len() == 160, "Ma trận G phải có 160 hàng");
        for row in &g {
            anyhow::ensure!(row.len() == 832, "Ma trận G phải có 832 cột");
        }
        Ok(Self { g })
    }

    pub fn encode(&self, random_key: &[u8]) -> Vec<u8> {
        debug_assert_eq!(random_key.len(), 20, "random_key phải đúng 160-bit (20 byte)");

        let mut key_bits = vec![0u8; 160];
        for i in 0..20 {
            for j in 0..8 {
                key_bits[i * 8 + j] = (random_key[i] >> (7 - j)) & 1;
            }
        }
        let mut codeword = vec![0u8; 832];
        for i in 0..832 {
            let mut sum = 0u32;
            for j in 0..160 {
                sum += (key_bits[j] as u32) * (self.g[j][i] as u32);
            }
            codeword[i] = (sum % 2) as u8;
        }
        codeword
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_systematic() {
        let encoder = LdpcEncoder::new("generator_matrix_G.txt").unwrap();
        // Key khác 0 — bắt buộc, vì key = 0 luôn cho codeword = 0 với MỌI
        // ma trận G (kể cả G sai hoàn toàn), không kiểm chứng được tính
        // systematic thật sự (đã chứng minh bằng thực nghiệm ở câu trả lời
        // trước — test kiểu cũ pass ngay cả trên ma trận G ngẫu nhiên).
        let key: Vec<u8> = (0..20u8).map(|i| i.wrapping_mul(37).wrapping_add(11)).collect();
        let codeword = encoder.encode(&key);
        let key_bits: Vec<u8> = (0..160)
            .map(|i| (key[i / 8] >> (7 - (i % 8))) & 1)
            .collect();
        assert_eq!(&codeword[..160], &key_bits[..]);
    }
}