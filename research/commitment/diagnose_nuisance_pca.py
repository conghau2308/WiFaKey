"""
diagnose_nuisance_pca.py

Chạy PCA trên vector hiệu số (diff = embedding_verify - embedding_enroll)
của các cặp genuine trong tập tune LFW.
Xác định xem nhiễu có cấu trúc low‑rank hay không.
"""

import os, sys, csv, numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)


# --- Loaders ---
def load_embedding(name, imagenum):
    path = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "embeddings_cache",
        f"{name}_{int(imagenum):04d}.npy",
    )
    return np.load(path)


def load_genuine_pairs(max_pairs=None):
    pairs_csv = os.path.join(
        _PROJECT_ROOT,
        "datasets",
        "processed",
        "labeled_faces_in_the_wild",
        "pairs",
        "tune_genuine.csv",
    )
    pairs = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                (
                    row["name_enroll"],
                    int(row["imagenum_enroll"]),
                    row["name_verify"],
                    int(row["imagenum_verify"]),
                )
            )
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def main():
    pairs = load_genuine_pairs()  # 881 cặp
    print(f"Số cặp genuine: {len(pairs)}")

    # Thu thập vector hiệu số
    diffs = []
    for name_e, img_e, name_v, img_v in pairs:
        emb_enroll = load_embedding(name_e, img_e)
        emb_verify = load_embedding(name_v, img_v)
        diff = emb_verify - emb_enroll  # (512,)
        diffs.append(diff)

    diffs = np.array(diffs)  # (N, 512)
    print(f"Shape ma trận hiệu số: {diffs.shape}")

    # Chuẩn hóa về mean=0 (thường đã gần 0)
    mean_diff = np.mean(diffs, axis=0)
    diffs_centered = diffs - mean_diff

    # PCA qua SVD (nhanh và chính xác)
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)

    # Tính tỷ lệ phương sai giải thích
    explained_variance = (S**2) / np.sum(S**2)
    cumulative = np.cumsum(explained_variance)

    print("\n=== KẾT QUẢ PCA TRÊN VECTOR HIỆU SỐ ===")
    print(f"Tổng phương sai: {np.sum(S**2):.4f}")
    print(f"\nTop 10 thành phần chính:")
    print(f"{'PC':<6} {'Giải thích (%)':<15} {'Tích lũy (%)':<15}")
    print("-" * 36)
    for i in range(min(10, len(S))):
        print(
            f"PC{i+1:<5} {explained_variance[i]*100:>14.4f} {cumulative[i]*100:>14.4f}"
        )

    # Nhận xét
    top5_cumsum = cumulative[4] if len(S) >= 5 else 1.0
    top10_cumsum = cumulative[9] if len(S) >= 10 else 1.0
    print(f"\nTop 5 PC giải thích: {top5_cumsum*100:.2f}% phương sai")
    print(f"Top 10 PC giải thích: {top10_cumsum*100:.2f}% phương sai")

    if top5_cumsum > 0.5:
        print("\n=> NHIỄU CÓ CẤU TRÚC LOW-RANK RẤT RÕ RỆT (top 5 PC > 50% variance).")
        print("   NAP hoặc điều chỉnh LLR theo ngữ cảnh rất hứa hẹn.")
    elif top10_cumsum > 0.5:
        print("\n=> NHIỄU CÓ CẤU TRÚC LOW-RANK VỪA PHẢI (top 10 PC > 50% variance).")
        print("   Có thể thử NAP với K=5-10 hoặc điều chỉnh LLR.")
    else:
        print("\n=> NHIỄU GẦN NHƯ ĐẲNG HƯỚNG (top 10 PC < 50% variance).")
        print(
            "   NAP khó có hiệu quả, nhưng điều chỉnh LLR theo ngữ cảnh vẫn có thể thử."
        )


if __name__ == "__main__":
    main()
