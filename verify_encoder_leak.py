"""
verify_encoder_leak.py

Kiểm tra 2 việc trên chính generator matrix G thật của hệ thống WiFaKey:

1. G có phải dạng SYSTEMATIC ([I_K | P] hoặc [P | I_K]) không?
   -> Nếu có, các vị trí lộ (do AND-mask-về-0 rồi XOR) rơi vào đúng dải
      đó sẽ lộ TRỰC TIẾP bit của key k, không cần giải gì cả.

2. Mô phỏng tấn công Information-Set Decoding (ISD) bằng khử Gauss
   trên GF(2): với các vị trí mask=0 (nên delta_i = c_i, đã biết),
   thử khôi phục lại k bằng cách giải hệ phương trình tuyến tính
   k @ G_sub = c_known (mod 2), với G_sub là submatrix của G ứng với
   các cột đã biết.

Cách chạy:
    python verify_encoder_leak.py --data-dir /path/to/wifakey_module/data \\
        --kappa 0.3125 --z 16 --trials 200

Yêu cầu: numpy. Thay --data-dir bằng đường dẫn thật tới thư mục chứa
BaseGraph_GM/LDPC_GM_BG2_16.txt trong project của bạn.
"""

import argparse
import os

import numpy as np


def load_gmatrix(data_dir: str, z: int) -> np.ndarray:
    fname_map = {
        16: "LDPC_GM_BG2_16.txt",
        3: "LDPC_GM_BG2_3.txt",
        10: "LDPC_GM_BG2_10.txt",
        6: "LDPC_GM_BG2_6.txt",
    }
    if z not in fname_map:
        raise ValueError(f"Z={z} không hợp lệ, chỉ hỗ trợ {list(fname_map)}")
    path = os.path.join(data_dir, "BaseGraph_GM", fname_map[z])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy {path} — kiểm tra lại --data-dir")
    G = np.loadtxt(path, dtype=int, delimiter=",")
    return G % 2


def check_systematic(G: np.ndarray):
    """Kiểm tra G có dạng [I_K | P] (đầu) hoặc [P | I_K] (cuối) không."""
    K, N = G.shape
    I_K = np.eye(K, dtype=int)

    front = G[:, :K] % 2
    is_systematic_front = np.array_equal(front, I_K)

    back = G[:, N - K :] % 2
    is_systematic_back = np.array_equal(back, I_K)

    return is_systematic_front, is_systematic_back


def gf2_rank(A: np.ndarray) -> int:
    """Rank của ma trận nhị phân trên GF(2) bằng khử Gauss."""
    A = A.copy() % 2
    rows, cols = A.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if A[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        A[[rank, pivot]] = A[[pivot, rank]]
        for r in range(rows):
            if r != rank and A[r, col] == 1:
                A[r] = (A[r] + A[rank]) % 2
        rank += 1
        if rank == rows:
            break
    return rank


def gf2_solve(A: np.ndarray, b: np.ndarray):
    """
    Giải A x = b (mod 2) bằng khử Gauss.
    A: (m, n), b: (m,). Trả về 1 nghiệm x (n,) nếu hệ có nghiệm, None nếu vô nghiệm.
    Nếu hệ thiếu rank (under-determined), trả về nghiệm riêng với các biến tự do = 0.
    """
    m, n = A.shape
    Aug = np.concatenate([A % 2, (b % 2).reshape(-1, 1)], axis=1)
    row = 0
    pivot_cols = []
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if Aug[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        Aug[[row, pivot]] = Aug[[pivot, row]]
        for r in range(m):
            if r != row and Aug[r, col] == 1:
                Aug[r] = (Aug[r] + Aug[row]) % 2
        pivot_cols.append(col)
        row += 1
        if row == m:
            break

    # Kiểm tra tính nhất quán (hàng toàn 0 nhưng vế phải = 1 -> vô nghiệm)
    for r in range(row, m):
        if Aug[r, :n].sum() == 0 and Aug[r, n] == 1:
            return None

    x = np.zeros(n, dtype=int)
    for i, col in enumerate(pivot_cols):
        x[col] = Aug[i, n]
    return x


def simulate_attack(G: np.ndarray, kappa: float, trials: int = 200, seed: int = 0):
    """
    Mô phỏng đúng pipeline enroll hiện tại:
        c = k @ G mod 2
        mask_r = (u >= kappa)          # 1 = giữ, 0 = che về 0
        b_masked = b_full & mask_r
        delta = b_masked XOR c

    Tại vị trí mask_r=0: delta_i = c_i (đã biết với attacker vì mask_r công khai).
    Attacker dùng các vị trí đó để thử giải ngược k qua Gaussian elimination.
    """
    rng = np.random.default_rng(seed)
    K, N = G.shape
    successes = 0
    avg_leaked = 0

    for _ in range(trials):
        k = rng.integers(0, 2, size=K)
        c = (k @ G) % 2

        u = rng.random(N)
        mask_r = (u >= kappa).astype(int)  # 1 = giữ, 0 = che

        b_full = rng.integers(
            0, 2, size=N
        )  # bit sinh trắc giả lập (attacker không biết)
        b_masked = b_full & mask_r
        delta = b_masked ^ c

        leaked_positions = np.where(mask_r == 0)[0]
        avg_leaked += len(leaked_positions)

        if len(leaked_positions) < K:
            continue  # trial này không đủ bit lộ để thử giải

        c_known = delta[leaked_positions]  # == c[leaked_positions]
        G_sub = G[:, leaked_positions] % 2  # (K, num_leaked)

        # Giải k @ G_sub = c_known (mod 2)  <=>  G_sub.T @ k.T = c_known.T
        A = G_sub.T  # (num_leaked, K)
        sol = gf2_solve(A, c_known)

        if sol is not None and np.array_equal((sol @ G) % 2, c):
            successes += 1

    print(f"\nSố trial: {trials}")
    print(
        f"Số vị trí lộ trung bình (mask=0): {avg_leaked / trials:.1f} / {N}  (kỳ vọng lý thuyết: {kappa*N:.1f})"
    )
    print(
        f"Số lần khôi phục ĐÚNG k qua Gaussian elimination: {successes} ({100*successes/trials:.1f}%)"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir", required=True, help="Đường dẫn tới wifakey_module/data"
    )
    ap.add_argument("--kappa", type=float, default=0.3125)
    ap.add_argument("--z", type=int, default=16, choices=[16, 3, 10, 6])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Đang load G_matrix (Z={args.z}) từ {args.data_dir} ...")
    G = load_gmatrix(args.data_dir, args.z)
    K, N = G.shape
    print(f"G shape: K={K}, N={N}")

    front, back = check_systematic(G)
    print(f"\nSystematic dạng [I_K | P] (K cột đầu = I_K)?  {front}")
    print(f"Systematic dạng [P | I_K] (K cột cuối = I_K)?  {back}")
    if front or back:
        print(
            "=> CẢNH BÁO: G là systematic — key k xuất hiện trực tiếp, không biến đổi,"
        )
        print(
            "   ở K vị trí cụ thể của codeword c. Nếu các vị trí đó bị mask=0, k bị lộ"
        )
        print("   THẲNG qua delta, không cần giải Gauss.")

    rank = gf2_rank(G)
    print(
        f"\nRank thực tế của G trên GF(2): {rank} / {K}"
        f"  ({'đủ hạng, full row rank' if rank == K else 'THIẾU HẠNG — kiểm tra lại file G'})"
    )

    simulate_attack(G, args.kappa, trials=args.trials, seed=args.seed)


if __name__ == "__main__":
    main()
