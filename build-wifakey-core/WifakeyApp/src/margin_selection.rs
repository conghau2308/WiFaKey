// file: src/margin_selection.rs

/// Chọn ra `k` chỉ số có giá trị margin lớn nhất.
///
/// # Arguments
/// * `margin` - Slice chứa các giá trị margin (f64).
/// * `k` - Số lượng phần tử cần chọn.
///
/// # Returns
/// Một vector chứa các chỉ số đã được sắp xếp tăng dần.
pub fn select_top_margin_indices(margin: &[f64], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, &f64)> = margin.iter().enumerate().collect();
    // Sắp xếp giảm dần theo giá trị margin (dùng `total_cmp` để so sánh f64).
    // Trong Python: idx = np.argpartition(-margin, k)[:k]; idx.sort()
    // Ở đây ta làm tương tự: lấy top k phần tử có margin lớn nhất, sau đó sắp xếp chỉ số.
    indexed.sort_by(|a, b| b.1.total_cmp(a.1)); // Sắp xếp giảm dần
    let mut indices: Vec<usize> = indexed.into_iter().take(k).map(|(i, _)| i).collect();
    indices.sort(); // Sắp xếp tăng dần chỉ số
    indices
}

/// Tạo reliability mask từ danh sách chỉ số.
///
/// # Arguments
/// * `indices` - Slice chứa các chỉ số được chọn.
/// * `total_len` - Tổng độ dài của mask.
///
/// # Returns
/// Một vector boolean (`bool`) đại diện cho mask, với `true` tại các vị trí được chọn.
pub fn build_mask(indices: &[usize], total_len: usize) -> Vec<bool> {
    let mut mask = vec![false; total_len];
    for &i in indices {
        if i < total_len {
            mask[i] = true;
        }
    }
    mask
}

/// Trích xuất danh sách chỉ số từ mask (dùng trong quá trình Verify).
///
/// # Arguments
/// * `mask` - Slice boolean đại diện cho mask.
///
/// # Returns
/// Một vector chứa các chỉ số có giá trị `true`.
pub fn indices_from_mask(mask: &[bool]) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_top_margin_indices() {
        let margin = vec![0.1, 0.9, 0.5, 0.8, 0.3];
        let indices = select_top_margin_indices(&margin, 3);
        assert_eq!(indices, vec![1, 2, 3]); // vị trí 1 (0.9), 2 (0.8), 3 (0.5) nhưng sort nên 1,2,3
    }

    #[test]
    fn test_build_and_indices_from_mask() {
        let indices = vec![1, 3, 5];
        let mask = build_mask(&indices, 10);
        assert_eq!(mask, vec![false, true, false, true, false, true, false, false, false, false]);
        let recovered = indices_from_mask(&mask);
        assert_eq!(recovered, indices);
    }
}
