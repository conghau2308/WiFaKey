//! Port của _nms() (Python) — bước 8 trong pipeline RetinaFace/SCRFD post-
//! processing. Không phụ thuộc file model — kiểm chứng được ngay bằng dữ
//! liệu tổng hợp, đã làm ở đây.

/// Non-Maximum Suppression — trả về chỉ số các box được giữ lại, ĐÚNG THỨ
/// TỰ như Python (giảm dần theo score, greedy loại các box chồng lấn).
pub fn nms(bboxes: &[[f64; 4]], scores: &[f64], thresh: f64) -> Vec<usize> {
    let n = bboxes.len();
    let areas: Vec<f64> = bboxes
        .iter()
        .map(|b| (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0))
        .collect();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).expect("score không được NaN"));

    let mut keep = Vec::new();
    let mut order = std::collections::VecDeque::from(order);

    while let Some(i) = order.pop_front() {
        keep.push(i);
        let mut remaining = std::collections::VecDeque::new();
        for &j in order.iter() {
            let xx1 = bboxes[i][0].max(bboxes[j][0]);
            let yy1 = bboxes[i][1].max(bboxes[j][1]);
            let xx2 = bboxes[i][2].min(bboxes[j][2]);
            let yy2 = bboxes[i][3].min(bboxes[j][3]);
            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;
            let iou = inter / (areas[i] + areas[j] - inter);
            if iou <= thresh {
                remaining.push_back(j);
            }
        }
        order = remaining;
    }

    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_reference_exact_order() {
        let bboxes = [
            [10.0, 10.0, 110.0, 110.0],
            [15.0, 15.0, 115.0, 115.0],
            [12.0, 8.0, 108.0, 112.0],
            [300.0, 300.0, 400.0, 400.0],
            [305.0, 295.0, 405.0, 395.0],
            [500.0, 500.0, 550.0, 550.0],
        ];
        let scores = [0.9, 0.75, 0.6, 0.95, 0.7, 0.4];

        let keep = nms(&bboxes, &scores, 0.4);

        assert_eq!(keep, vec![3, 0, 5], "Phải khớp CHÍNH XÁC cả thứ tự lẫn nội dung với Python");
    }
}