//! Port của align_face() (Python) — dùng thuật toán Umeyama (1991) để ước
//! lượng similarity transform (xoay + scale + tịnh tiến) từ 5 keypoints
//! khuôn mặt sang 5 điểm tham chiếu chuẩn InsightFace.

use nalgebra::{Matrix2, Vector2};

pub const REFERENCE_LANDMARKS: [[f64; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// Trả về `[[a, b, tx], [c, d, ty]]` — dùng trực tiếp cho warpAffine.
pub fn estimate_similarity_transform(src: &[[f64; 2]; 5], dst: &[[f64; 2]; 5]) -> [[f64; 3]; 2] {
    let n = 5.0_f64;

    let mu_src = mean_point(src);
    let mu_dst = mean_point(dst);

    let src_demean: Vec<Vector2<f64>> = src
        .iter()
        .map(|p| Vector2::new(p[0] - mu_src[0], p[1] - mu_src[1]))
        .collect();
    let dst_demean: Vec<Vector2<f64>> = dst
        .iter()
        .map(|p| Vector2::new(p[0] - mu_dst[0], p[1] - mu_dst[1]))
        .collect();

    let mut a = Matrix2::zeros();
    for i in 0..5 {
        a += dst_demean[i] * src_demean[i].transpose();
    }
    a /= n;

    let det_a = a.determinant();
    let mut d = Vector2::new(1.0, 1.0);
    if det_a < 0.0 {
        d[1] = -1.0;
    }

    let svd = a.svd(true, true);
    let u = svd.u.expect("SVD phải tính được U");
    let v_t = svd.v_t.expect("SVD phải tính được V^T");
    let singular_values = svd.singular_values;

    let eps = 1e-10 * singular_values[0].max(1.0);
    let rank = singular_values.iter().filter(|&&s| s > eps).count();

    let det_u = u.determinant();
    let det_vt = v_t.determinant();

    let rotation = if rank == 0 {
        panic!("rank 0 — 5 điểm suy biến, không thể ước lượng transform");
    } else if rank == 1 {
        if det_u * det_vt > 0.0 {
            u * v_t
        } else {
            let mut d2 = d;
            let saved = d2[1];
            d2[1] = -1.0;
            let r = u * Matrix2::from_diagonal(&d2) * v_t;
            d2[1] = saved;
            r
        }
    } else {
        u * Matrix2::from_diagonal(&d) * v_t
    };

    let var_src: f64 = {
        let var_x: f64 = src_demean.iter().map(|p| p.x * p.x).sum::<f64>() / n;
        let var_y: f64 = src_demean.iter().map(|p| p.y * p.y).sum::<f64>() / n;
        var_x + var_y
    };

    let scale = (singular_values.dot(&d)) / var_src;

    let r_scaled = rotation * scale;
    let mu_src_vec = Vector2::new(mu_src[0], mu_src[1]);
    let mu_dst_vec = Vector2::new(mu_dst[0], mu_dst[1]);
    let translation = mu_dst_vec - r_scaled * mu_src_vec;

    [
        [r_scaled[(0, 0)], r_scaled[(0, 1)], translation[0]],
        [r_scaled[(1, 0)], r_scaled[(1, 1)], translation[1]],
    ]
}

fn mean_point(points: &[[f64; 2]; 5]) -> [f64; 2] {
    let mut sx = 0.0;
    let mut sy = 0.0;
    for p in points {
        sx += p[0];
        sy += p[1];
    }
    [sx / 5.0, sy / 5.0]
}

pub fn warp_affine(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    channels: usize,
    m: &[[f64; 3]; 2],
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    let a = m[0][0]; let b = m[0][1]; let tx = m[0][2];
    let c = m[1][0]; let d = m[1][1]; let ty = m[1][2];

    let det = a * d - b * c;
    let inv_a = d / det; let inv_b = -b / det;
    let inv_c = -c / det; let inv_d = a / det;

    let mut out = vec![0u8; dst_w * dst_h * channels];

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let x = dx as f64; let y = dy as f64;
            let sx = inv_a * (x - tx) + inv_b * (y - ty);
            let sy = inv_c * (x - tx) + inv_d * (y - ty);

            let x0 = sx.floor(); let y0 = sy.floor();
            let wx = sx - x0; let wy = sy - y0;
            let x0i = x0 as i64; let y0i = y0 as i64;

            for ch in 0..channels {
                let sample = |xi: i64, yi: i64| -> f64 {
                    if xi < 0 || yi < 0 || xi >= src_w as i64 || yi >= src_h as i64 {
                        0.0
                    } else {
                        src[(yi as usize * src_w + xi as usize) * channels + ch] as f64
                    }
                };
                let p00 = sample(x0i, y0i);
                let p01 = sample(x0i + 1, y0i);
                let p10 = sample(x0i, y0i + 1);
                let p11 = sample(x0i + 1, y0i + 1);
                let top = p00 * (1.0 - wx) + p01 * wx;
                let bottom = p10 * (1.0 - wx) + p11 * wx;
                let value = (top * (1.0 - wy) + bottom * wy).round().clamp(0.0, 255.0) as u8;
                out[(dy * dst_w + dx) * channels + ch] = value;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_skimage_reference_output() {
        let src = [
            [140.1999969482422, 165.8000030517578],
            [210.5, 160.10000610351562],
            [175.0, 210.3000030517578],
            [150.6999969482422, 250.89999389648438],
            [205.3000030517578, 248.1999969482422],
        ];
        let dst = REFERENCE_LANDMARKS;

        let m = estimate_similarity_transform(&src, &dst);

        let expected = [
            [0.48327454764448735, -0.01886852473778418, -25.28755436974798],
            [0.01886852473778418, 0.48327454764448735, -31.493315140254836],
        ];

        let tolerance = 1e-4;
        for i in 0..2 {
            for j in 0..3 {
                let diff = (m[i][j] - expected[i][j]).abs();
                assert!(diff < tolerance, "M[{i}][{j}] = {} khác expected {} (diff {diff})", m[i][j], expected[i][j]);
            }
        }
    }

    #[test]
    fn warp_affine_matches_python_end_to_end() {
        let src_bgr = std::fs::read("face_detection_test_image.bin").unwrap();
        assert_eq!(src_bgr.len(), 250 * 250 * 3);
        let expected_rgb = std::fs::read("aligned_face_112.bin").unwrap();
        assert_eq!(expected_rgb.len(), 112 * 112 * 3);

        let kps = [
            [99.2371597290039, 113.78105163574219],
            [144.4002227783203, 113.17049407958984],
            [125.83100891113281, 135.90870666503906],
            [100.68347930908203, 157.98382568359375],
            [144.6379852294922, 157.2723846435547],
        ];

        let m = estimate_similarity_transform(&kps, &REFERENCE_LANDMARKS);
        let aligned_bgr = warp_affine(&src_bgr, 250, 250, 3, &m, 112, 112);

        let mut aligned_rgb = vec![0u8; aligned_bgr.len()];
        for i in 0..(112 * 112) {
            aligned_rgb[i * 3] = aligned_bgr[i * 3 + 2];
            aligned_rgb[i * 3 + 1] = aligned_bgr[i * 3 + 1];
            aligned_rgb[i * 3 + 2] = aligned_bgr[i * 3];
        }

        let mut max_diff = 0i32;
        for i in 0..aligned_rgb.len() {
            let diff = (aligned_rgb[i] as i32 - expected_rgb[i] as i32).abs();
            max_diff = max_diff.max(diff);
        }
        println!("warp_affine max_diff = {max_diff}");
        assert!(max_diff <= 2, "max_diff quá lớn: {max_diff}");
    }
}