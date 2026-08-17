use std::fs;

/// Cấu trúc chứa bảng tra cứu Empirical LLR.
pub struct EmpiricalLlr {
    margin_bp: Vec<f64>,  // các điểm chia margin
    p_bp: Vec<f64>,       // xác suất lỗi tại các điểm chia
    eps: f64,             // giá trị epsilon để tránh log(0)
}

impl EmpiricalLlr {
    /// Khởi tạo từ file text xuất ra từ Python.
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut lines = content.lines();
        
        // Dòng đầu là số breakpoints
        let n: usize = lines.next().unwrap().parse()?;
        
        let mut margin_bp = Vec::with_capacity(n);
        let mut p_bp = Vec::with_capacity(n);
        
        for line in lines {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                margin_bp.push(parts[0].parse::<f64>()?);
                p_bp.push(parts[1].parse::<f64>()?);
            }
        }
        
        Ok(Self {
            margin_bp,
            p_bp,
            eps: 1e-6,
        })
    }
    
    /// Tra cứu LLR magnitude từ margin.
    /// Thực hiện nội suy tuyến tính giữa các breakpoint.
    pub fn margin_to_llr_magnitude(&self, margin: f64) -> f64 {
        // Tìm vị trí của margin trong mảng breakpoint (nội suy tuyến tính)
        let n = self.margin_bp.len();
        
        // Nếu margin nằm ngoài khoảng, trả về giá trị biên
        if margin <= self.margin_bp[0] {
            let p = self.p_bp[0];
            let p = p.max(self.eps).min(0.5 - self.eps);
            return ((1.0 - p) / p).ln();
        }
        if margin >= self.margin_bp[n - 1] {
            let p = self.p_bp[n - 1];
            let p = p.max(self.eps).min(0.5 - self.eps);
            return ((1.0 - p) / p).ln();
        }
        
        // Tìm khoảng chứa margin
        let mut idx = 0;
        for i in 0..n - 1 {
            if margin >= self.margin_bp[i] && margin <= self.margin_bp[i + 1] {
                idx = i;
                break;
            }
        }
        
        // Nội suy tuyến tính p
        let t = (margin - self.margin_bp[idx]) / (self.margin_bp[idx + 1] - self.margin_bp[idx]);
        let p = self.p_bp[idx] + t * (self.p_bp[idx + 1] - self.p_bp[idx]);
        let p = p.max(self.eps).min(0.5 - self.eps);
        
        ((1.0 - p) / p).ln()
    }
    
    /// Tính LLR đầy đủ: sign * magnitude.
    /// noisy_bit: true nếu bit noisy = 1, false nếu = 0.
    pub fn compute_llr(&self, noisy_bit: bool, margin: f64) -> f64 {
        let sign = if noisy_bit { 1.0 } else { -1.0 };
        sign * self.margin_to_llr_magnitude(margin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_llr_monotonic() {
        let llr = EmpiricalLlr::new("empirical_llr_table.txt").unwrap();
        // Margin càng lớn -> |LLR| càng lớn
        let m1 = llr.margin_to_llr_magnitude(0.01);
        let m2 = llr.margin_to_llr_magnitude(0.1);
        assert!(m2 > m1, "LLR magnitude should increase with margin");
    }
}