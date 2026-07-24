"""
06_splithalf_reliability_check.py

Kiểm soát cho 05_dimension_separation_audit.py: rho cross-fold trung bình
đo được là -0.001 (~0). Trước khi kết luận đó là bằng chứng THIÊN LỆCH
NHÂN KHẨU HỌC, cần loại trừ khả năng thay thế: separation_d vốn dĩ là tín
hiệu quá nhiễu (std giữa các dim chỉ gấp ~1.1-1.3 lần SE đo được) nên rho
sẽ gần 0 với BẤT KỲ cách chia mẫu nào, kể cả chia ngẫu nhiên trong cùng 1
fold — không liên quan gì đến nhân khẩu học.

PHƯƠNG PHÁP: với mỗi fold, chia ngẫu nhiên genuine pairs thành 2 nửa độc
lập (và impostor pairs thành 2 nửa độc lập riêng), tính lại separation_d
cho mỗi nửa, đo Spearman rho giữa 2 nửa CÙNG 1 fold (split-half rho).

DIỄN GIẢI:
  - split-half rho cũng ~0  -> separation_d vốn không ổn định (tín hiệu
    quá yếu so với nhiễu đo lường), không phải do khác biệt nhân khẩu
    học. Đây khớp với phát hiện trước đó (interior/exterior BER chỉ
    1.31x) — cùng một kết luận: quantizer hiện tại cho tín hiệu độ tin
    cậy/phân tách rất yếu nói chung.
  - split-half rho cao (vd >0.5) NHƯNG cross-fold rho ~0 -> bằng chứng
    thiên lệch nhân khẩu học THẬT, vì cùng 1 fold thì ổn định, khác
    fold thì không.

Cách chạy:
    python scripts/06_splithalf_reliability_check.py
"""

import os
import sys
import csv
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.insert(0, _PROJECT_ROOT)

from wifakey_module.wifakey_lib import utils

DATASET_NAME = "demogpairs"
DATA_DIR = os.path.join(_PROJECT_ROOT, "wifakey_module", "data")
M_MATRIX_NPY = os.path.join(DATA_DIR, "M_matrix.npy")
INTERVALS_NPY = os.path.join(DATA_DIR, "binarization_intervals.npy")

PAIRS_DIR = os.path.join(_PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "pairs")
GENUINE_CSV = os.path.join(PAIRS_DIR, "audit_genuine.csv")
IMPOSTOR_SAMEFOLD_CSV = os.path.join(PAIRS_DIR, "audit_impostor_samefold.csv")
EMBEDDINGS_CACHE_DIR = os.path.join(
    _PROJECT_ROOT, "datasets", "processed", DATASET_NAME, "embeddings_cache"
)

OUT_DIR = os.path.join(_PROJECT_ROOT, "experiments", "out_dim_separation_audit")
os.makedirs(OUT_DIR, exist_ok=True)

N_DIMS_BASELINE_BUDGET = 277
RNG_SEED = 42
N_SPLITS = 5  # lặp lại nhiều lần chia ngẫu nhiên, tránh 1 lần chia may rủi

M_matrix = np.load(M_MATRIX_NPY)
intervals = np.load(INTERVALS_NPY)
N_THR = np.asarray(intervals).reshape(-1).size
D_DIMS = M_matrix.shape[1]


def real_binarize(emb: np.ndarray) -> np.ndarray:
    projected = np.dot(np.asarray(emb, dtype=np.float64), M_matrix)
    return (
        utils.lssc_binary(projected[None, :], interval=intervals)
        .flatten()
        .astype(np.uint8)
    )


_embedding_cache = {}


def load_embedding(cache_filename: str) -> np.ndarray:
    if cache_filename not in _embedding_cache:
        path = os.path.join(EMBEDDINGS_CACHE_DIR, cache_filename)
        _embedding_cache[cache_filename] = np.load(path)
    return _embedding_cache[cache_filename]


def load_pairs_by_fold():
    pairs_by_fold = defaultdict(lambda: {"genuine": [], "impostor": []})
    with open(GENUINE_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs_by_fold[row["fold"]]["genuine"].append(
                (row["cache_filename_1"], row["cache_filename_2"])
            )
    with open(IMPOSTOR_SAMEFOLD_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pairs_by_fold[row["fold"]]["impostor"].append(
                (row["cache_filename_1"], row["cache_filename_2"])
            )
    return pairs_by_fold


def per_dimension_ber(pairs: list) -> np.ndarray:
    if not pairs:
        return np.full(D_DIMS, np.nan)
    flips_sum = np.zeros(D_DIMS * N_THR, dtype=np.float64)
    for f1, f2 in pairs:
        b1 = real_binarize(load_embedding(f1))
        b2 = real_binarize(load_embedding(f2))
        flips_sum += (b1 != b2).astype(np.float64)
    return (flips_sum / len(pairs)).reshape(D_DIMS, N_THR).mean(axis=1)


def split_half_separation(genuine_pairs, impostor_pairs, rng):
    """Chia ngẫu nhiên genuine và impostor (độc lập) thành 2 nửa,
    trả về (separation_half_A, separation_half_B)."""
    g_idx = rng.permutation(len(genuine_pairs))
    i_idx = rng.permutation(len(impostor_pairs))

    g_a = [genuine_pairs[k] for k in g_idx[: len(g_idx) // 2]]
    g_b = [genuine_pairs[k] for k in g_idx[len(g_idx) // 2 :]]
    i_a = [impostor_pairs[k] for k in i_idx[: len(i_idx) // 2]]
    i_b = [impostor_pairs[k] for k in i_idx[len(i_idx) // 2 :]]

    sep_a = per_dimension_ber(i_a) - per_dimension_ber(g_a)
    sep_b = per_dimension_ber(i_b) - per_dimension_ber(g_b)
    return sep_a, sep_b, len(g_a), len(g_b), len(i_a), len(i_b)


def jaccard_topk(sep_a, sep_b, k=N_DIMS_BASELINE_BUDGET):
    top_a = set(np.argsort(-sep_a)[:k].tolist())
    top_b = set(np.argsort(-sep_b)[:k].tolist())
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return inter / union if union else 0.0, inter


def main():
    pairs_by_fold = load_pairs_by_fold()
    rng = np.random.default_rng(RNG_SEED)

    # Kỳ vọng chance-level cho tham chiếu (K=277 > D/2=256):
    chance_overlap_frac = (N_DIMS_BASELINE_BUDGET**2 / D_DIMS) / N_DIMS_BASELINE_BUDGET
    print(
        f"[tham chiếu] Overlap kỳ vọng nếu 2 ranking ĐỘC LẬP HOÀN TOÀN: "
        f"{chance_overlap_frac:.1%} (do K={N_DIMS_BASELINE_BUDGET} > D/2={D_DIMS/2:.0f})\n"
    )

    all_rhos = defaultdict(list)
    all_jaccards = defaultdict(list)

    for fold_name, pairs in pairs_by_fold.items():
        genuine_pairs = pairs["genuine"]
        impostor_pairs = pairs["impostor"]
        if len(genuine_pairs) < 20 or len(impostor_pairs) < 20:
            print(
                f"[{fold_name}] *** Quá ít pairs ({len(genuine_pairs)} genuine / "
                f"{len(impostor_pairs)} impostor) để chia đôi đáng tin cậy — bỏ qua. ***"
            )
            continue

        print(
            f"[{fold_name}] genuine={len(genuine_pairs)}  impostor={len(impostor_pairs)}"
        )
        for split_i in range(N_SPLITS):
            sep_a, sep_b, ng_a, ng_b, ni_a, ni_b = split_half_separation(
                genuine_pairs, impostor_pairs, rng
            )
            rho, p = spearmanr(sep_a, sep_b)
            jac, overlap_n = jaccard_topk(sep_a, sep_b)
            all_rhos[fold_name].append(rho)
            all_jaccards[fold_name].append(jac)
            print(
                f"    split {split_i+1}/{N_SPLITS}: "
                f"n_genuine=({ng_a},{ng_b}) n_impostor=({ni_a},{ni_b})  "
                f"rho={rho:+.3f} (p={p:.1e})  "
                f"overlap={overlap_n}/{N_DIMS_BASELINE_BUDGET} ({jac:.3f} jaccard)"
            )

    print("\n" + "=" * 70)
    print("TỔNG KẾT SPLIT-HALF RELIABILITY (trung bình qua các fold & split)")
    print("=" * 70)
    fold_avg_rhos = {f: float(np.mean(v)) for f, v in all_rhos.items()}
    for f, avg_rho in fold_avg_rhos.items():
        print(
            f"  {f:16s}: avg split-half rho = {avg_rho:+.3f} "
            f"(so với cross-fold rho ≈ -0.001 đã đo ở 05)"
        )

    overall_avg = float(np.mean([v for vs in all_rhos.values() for v in vs]))
    print(f"\nTrung bình toàn bộ split-half rho: {overall_avg:+.3f}")

    print("\nDIỄN GIẢI:")
    if overall_avg < 0.2:
        print(
            "  -> Split-half rho CŨNG gần 0, tương đương cross-fold rho (~0).\n"
            "     Kết luận: separation_d vốn không ổn định ngay cả trong CÙNG\n"
            "     1 fold -> NO-GO ở 05 chủ yếu do TÍN HIỆU QUÁ NHIỄU (khớp phát\n"
            "     hiện interior/exterior BER=1.31x trước đó), KHÔNG PHẢI bằng\n"
            "     chứng đủ mạnh cho thiên lệch nhân khẩu học cụ thể. Nên viết\n"
            "     luận văn theo hướng: 'separation-based dimension selection\n"
            "     không khả thi với quantizer hiện tại, bất kể nhóm nào' —\n"
            "     đây vẫn là kết quả khoa học có giá trị (âm tính, có kiểm chứng)."
        )
    elif overall_avg < 0.5:
        print(
            "  -> Split-half rho cao hơn cross-fold rho nhưng không mạnh.\n"
            "     Tín hiệu có tồn tại nhưng yếu; cần thêm dữ liệu (nhiều ảnh/\n"
            "     identity hơn) trước khi kết luận chắc chắn theo hướng nào."
        )
    else:
        print(
            "  -> Split-half rho CAO trong khi cross-fold rho ~0 -> đây là\n"
            "     bằng chứng THIÊN LỆCH NHÂN KHẨU HỌC thật, không phải nhiễu.\n"
            "     separation_d ổn định trong cùng nhóm nhưng đổi hẳn giữa các\n"
            "     nhóm -> kết luận NO-GO ở 05 đứng vững, và đây là một phát\n"
            "     hiện đáng giá trị khoa học cao cho luận văn (fairness/security)."
        )

    np.savez(
        os.path.join(OUT_DIR, "splithalf_reliability.npz"),
        fold_avg_rhos=np.array(list(fold_avg_rhos.items()), dtype=object),
        overall_avg_rho=overall_avg,
    )


if __name__ == "__main__":
    main()
