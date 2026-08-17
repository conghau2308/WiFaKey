"""
diagnostic_bitlevel_list_decoding.py

Muc dich: do can tren ly thuyet cua GMR dat duoc neu ap dung bit-level list
decoding (huong E) o domain 160-bit info, TRUOC KHI viet engine combinatorial
day du. Khong sua SecureWiFaKeyHandler / v1_selection_puncturing.py -- ke
thua doc lap, expose them ground-truth (random_key) va llr_info de do,
giong pattern v1_reduced_key_length.py.

Cau hoi tra loi: "Neu chon L bit |LLR| thap nhat trong 160 bit info, bao
nhieu % ca fail hien tai co TOAN BO vi tri loi that nam gon trong L bit do?"
-- day la dieu kien CAN (khong phai du) de list decoding weight-order tim ra
dung dap an, vi neu loi nam ngoai L bit thi khong to hop lat nao trong L bit
cuu duoc. Do truoc de biet huong E co dang viet engine combinatorial hay
khong, KHONG can chay bat ky phep thu to hop nao (O(1) moi ca, re).

LUU Y: day la chan doan dung random_key lam oracle (khong dung trong he
thong that) -- chi hop le trong moi truong test co ground-truth, giong het
cach reduced_key_128 da do con so 56.3%/61.4% truoc do.

Phat hien quan trong lam script nay don gian: self.decoder_output cua
WiFaKeyHandler da la LLR DAY DU cho toan bo 832-bit codeword (channel LLR +
extrinsic sau 25 vong Neural-MS), khong phai chi hard-decision. Vi G la
systematic [I_160|P], 160 bit info nam dung o dau mang LLR nay. Nghia la
llr_info = y_pred_llr.flatten()[:160] la MIEN PHI -- khong can forward pass
them, khong can sua decoder.
"""

import argparse
import csv
import hashlib
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from wifakey_module.wifakey_lib import Modulation

# --- Loader: TÁI DÙNG đúng convention của run_single_config.py, không viết
# lại logic mới --- (name_enroll/imagenum_enroll/name_verify/imagenum_verify
# trong CSV, file cache đặt tên "{name}_{imagenum:04d}.npy")


def load_embedding(cache_dir: str, name: str, imagenum) -> np.ndarray:
    return np.load(os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy"))


def load_pairs(pairs_csv: str, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def genuine_pairs_iter(pairs_csv: str, cache_dir: str, max_pairs=None):
    """
    Yield (feature_enroll, feature_verify) -- embedding AdaFace 512-dim thô,
    CHƯA binarize -- đúng input mà enroll_with_ground_truth()/verify_with_llr()
    cần (chúng tự gọi self._binarize_full() bên trong).

    pairs_csv: ví dụ ".../labeled_faces_in_the_wild/pairs/tune_genuine.csv"
               hoặc ".../cplfw/pairs/select_genuine.csv"
    cache_dir: ví dụ ".../labeled_faces_in_the_wild/embeddings_cache"
    """
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            yield e1, e2
        except Exception as e:
            print(f"  [WARN] lỗi load pair ({row}): {e}", file=sys.stderr)


class ListDecodingDiagnosticHandler(SecureWiFaKeyHandler):
    """
    enroll_with_ground_truth(): giong enroll() cua lop cha, nhung tra them
    random_key (KHONG dung trong he thong that, chi de do).

    verify_with_llr(): giong verify() cua lop cha, nhung tra them llr_info
    (160-bit) thay vi chi True/False.
    """

    def enroll_with_ground_truth(self, feature_vector_float: np.ndarray):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)

        rng = np.random.default_rng()
        selection_indices = rng.choice(
            len(b_full), size=self.feature_length, replace=False
        )
        selection_indices.sort()

        selection_mask = np.zeros(len(b_full), dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected = b_full[selection_indices]

        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)

        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()

        # Tra them random_key lam oracle -- KHONG dung trong production,
        # chi hop le vi day la moi truong test co ground-truth.
        return helper_data, selection_mask, key_hash, random_key.flatten()

    def verify_with_llr(
        self, feature_vector_float: np.ndarray, helper_data, selection_mask
    ):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        selection_indices = np.where(selection_mask == 1)[0]
        b_selected = b_full[selection_indices]
        y_noisy_bits = np.logical_xor(b_selected, helper_data)

        y_llr = (
            Modulation.BPSK(y_noisy_bits)
            .astype(np.float32)
            .reshape((1, self.N, self.Z))
        )
        y_pred_llr = self.sess.run(self.decoder_output, feed_dict={self.xa: y_llr})
        y_pred_llr = y_pred_llr.flatten()  # 832-bit LLR, systematic order

        decoded_codeword = (y_pred_llr > 0).astype(int)
        reconstructed_key = decoded_codeword[: self.key_length]
        llr_info = y_pred_llr[: self.key_length]  # 160-bit info LLR, mien phi

        return reconstructed_key, llr_info


def run_diagnostic(handler, genuine_pairs_iter, L_values=(16, 18, 20, 24, 32)):
    """
    genuine_pairs_iter: iterator yield (feature_enroll, feature_verify) --
    cam vao ham load-pair hien co cua ban (giong cach run_single_config.py
    doc LFW/CPLFW pairs), khong viet lai logic load dataset o day.

    Vi day chi la phep kiem tra "error_positions co la tap con cua L bit
    |LLR| thap nhat khong" (O(1) per case, khong thu to hop nao), co the
    chay tren FULL dataset (881 cap LFW / 1694 cap CPLFW) ma khong lo ve
    chi phi tinh toan -- khac han engine combinatorial that su.
    """
    n_total = 0
    n_fail = 0
    coverable = {L: 0 for L in L_values}
    error_count_of_fail = []

    for feature_enroll, feature_verify in genuine_pairs_iter:
        n_total += 1

        helper_data, selection_mask, key_hash, random_key = (
            handler.enroll_with_ground_truth(feature_enroll)
        )
        reconstructed_key, llr_info = handler.verify_with_llr(
            feature_verify, helper_data, selection_mask
        )

        if np.array_equal(reconstructed_key, random_key):
            continue  # da pass o fast path (hash check binh thuong)

        n_fail += 1
        error_positions = set(np.where(reconstructed_key != random_key)[0].tolist())
        error_count_of_fail.append(len(error_positions))

        # Tang dan do tin cay -- |LLR| thap nhat dung dau
        abs_llr_order = np.argsort(np.abs(llr_info))

        for L in L_values:
            candidate_positions = set(abs_llr_order[:L].tolist())
            if error_positions.issubset(candidate_positions):
                coverable[L] += 1

    print(f"Tong cap genuine: {n_total}")
    print(f"So ca fail (fast path): {n_fail} ({100*n_fail/max(n_total,1):.2f}%)")
    print(f"GMR hien tai (fast path only): {100*(n_total-n_fail)/max(n_total,1):.2f}%")
    if error_count_of_fail:
        print(
            f"So bit loi trung binh/ca fail: {np.mean(error_count_of_fail):.2f} "
            f"(median: {np.median(error_count_of_fail):.1f})"
        )
    print()
    print("Can tren GMR neu ap dung list decoding weight-order tai tung L:")
    print("(dieu kien CAN, chua tinh chi phi combinatorial thuc te)")
    for L in L_values:
        rescued = coverable[L]
        upper_bound_gmr = 100 * (n_total - n_fail + rescued) / max(n_total, 1)
        rescue_rate = 100 * rescued / n_fail if n_fail else 0.0
        print(
            f"  L={L:>3}: cuu duoc {rescued}/{n_fail} ca fail "
            f"({rescue_rate:.2f}%) -> GMR can tren = {upper_bound_gmr:.2f}%"
        )

    return {
        "n_total": n_total,
        "n_fail": n_fail,
        "coverable": coverable,
        "error_count_of_fail": error_count_of_fail,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--wifakey-data-dir", default=None)
    ap.add_argument("--pairs-dir", default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--dataset-folder",
        default="labeled_faces_in_the_wild",
        help="Tên thư mục dưới datasets/processed/. LFW: "
        "labeled_faces_in_the_wild (mac dinh). CPLFW: cplfw.",
    )
    ap.add_argument(
        "--tier",
        default="tune",
        help="Tien to file CSV: tune_genuine.csv (LFW) hoac "
        "select_genuine.csv (CPLFW, khong co tune_*).",
    )
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument(
        "--force-cpu",
        action="store_true",
        help="Ep chay CPU du co GPU (tranh OOM VRAM)",
    )
    ap.add_argument(
        "--L-values",
        type=int,
        nargs="+",
        default=[16, 18, 20, 24, 32],
    )
    args = ap.parse_args()

    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    root = os.path.abspath(args.project_root)
    data_dir = args.wifakey_data_dir or os.path.join(root, "wifakey_module", "data")
    pairs_dir = args.pairs_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "pairs"
    )
    cache_dir = args.cache_dir or os.path.join(
        root, "datasets", "processed", args.dataset_folder, "embeddings_cache"
    )
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")
    pairs_csv = os.path.join(pairs_dir, f"{args.tier}_genuine.csv")

    print(f"Doc genuine pairs tu: {pairs_csv}")
    print(f"Cache embedding tu : {cache_dir}")

    handler = ListDecodingDiagnosticHandler(
        data_path=data_dir, weights_path=weights_path, biases_path=biases_path
    )

    pairs_iter = genuine_pairs_iter(pairs_csv, cache_dir, args.max_pairs)
    run_diagnostic(handler, pairs_iter, L_values=tuple(args.L_values))


if __name__ == "__main__":
    main()
