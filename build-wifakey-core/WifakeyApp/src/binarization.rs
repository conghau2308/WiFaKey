// file: src/binarization.rs

/// Kết quả của quá trình binarize.
#[derive(Debug, Clone)]
pub struct BinarizationResult {
    pub bits: Vec<u8>,
    pub margin: Vec<f64>,
}

/// Binarize embedding đã biến đổi thành 1536 bit và margin.
///
/// # Arguments
/// * `feature_vector` - Vector embedding đã qua BioHashing (512 chiều).
/// * `m_matrix` - Ma trận gốc WiFaKey (512 x 512), row-major.
/// * `intervals` - Các ngưỡng thermometer code (3 giá trị).
pub fn binarize(
    feature_vector: &[f64],
    m_matrix: &[f64],
    intervals: &[f64],
) -> BinarizationResult {
    let dim = feature_vector.len();
    assert_eq!(dim, 512, "Feature vector must be 512-dimensional");
    assert_eq!(intervals.len(), 3, "Thermometer code requires exactly 3 thresholds");
    assert_eq!(m_matrix.len(), dim * dim, "M_matrix must be square");

    // Bước 1: Chiếu feature vector qua M_matrix
    let mut projected = vec![0.0; dim];
    for i in 0..dim {
        let mut sum = 0.0;
        for j in 0..dim {
            sum += m_matrix[i * dim + j] * feature_vector[j];
        }
        projected[i] = sum;
    }

    // Bước 2: Sắp xếp ngưỡng giảm dần (giống code gốc Python)
    // Python: rev_thr = np.sort(intervals)[::-1]
    let mut rev_thr = intervals.to_vec();
    rev_thr.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut bits = vec![0u8; 1536];
    let mut margin = vec![0.0f64; 1536];

    // Bước 3: Binarize từng chiều, gán bit và tính margin
    for d in 0..dim {
        let val = projected[d];
        for t_idx in 0..3 {
            let thr = rev_thr[t_idx];
            let pos = d * 3 + t_idx;
            bits[pos] = if val >= thr { 1 } else { 0 };
            margin[pos] = (val - thr).abs();
        }
    }

    BinarizationResult { bits, margin }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Intervals thực tế từ file binarization_intervals.npy
    fn real_intervals() -> Vec<f64> {
        vec![-0.029636395351665956, 0.0001636046483338635, 0.03006360464833386]
    }

    #[test]
    fn test_binarize_output_size() {
        let intervals = real_intervals();
        let feature = vec![0.0; 512];
        let m_matrix = vec![0.0; 512 * 512];
        let result = binarize(&feature, &m_matrix, &intervals);
        assert_eq!(result.bits.len(), 1536);
        assert_eq!(result.margin.len(), 1536);
    }

    #[test]
    fn test_binarize_identity_matrix_zero_feature() {
        // Với M là ma trận đơn vị và feature = 0, projected = 0.
        // rev_thr = [0.03006, 0.00016, -0.0296]
        // bit0 (0 >= 0.03006) = 0, margin0 = 0.03006
        // bit1 (0 >= 0.00016) = 0, margin1 = 0.00016
        // bit2 (0 >= -0.0296) = 1, margin2 = 0.0296
        let intervals = real_intervals();
        let feature = vec![0.0; 512];
        let mut m_matrix = vec![0.0; 512 * 512];
        for i in 0..512 {
            m_matrix[i * 512 + i] = 1.0;
        }
        let result = binarize(&feature, &m_matrix, &intervals);
        // Kiểm tra 3 bit đầu tiên (chiều 0)
        assert_eq!(result.bits[0], 0);
        assert_eq!(result.bits[1], 0);
        assert_eq!(result.bits[2], 1);
        // Kiểm tra margin (sai số 1e-10)
        assert!((result.margin[0] - 0.03006360464833386).abs() < 1e-10);
        assert!((result.margin[1] - 0.0001636046483338635).abs() < 1e-10);
        assert!((result.margin[2] - 0.029636395351665956).abs() < 1e-10);
    }
}