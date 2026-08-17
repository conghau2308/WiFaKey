"""
benchmark_single_config.py (ĐẦY ĐỦ CÁC PHƯƠNG PHÁP)

Chạy benchmark cho MỘT cấu hình duy nhất.
Hỗ trợ:
  --selection random|margin
  --llr bpsk|empirical
  --multi K sigma (K=0 để tắt)
  --oracle-lda PATH (tùy chọn)
  --neural-model PATH --neural-layers N --neural-units M --neural-activation relu|leaky_relu|swish
  --dataset lfw|cplfw
  --tier tune|select
  --max-pairs N
  --output file.json
"""

import argparse, csv, hashlib, json, os, sys, numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from research.commitment.v1_selection_puncturing import SecureWiFaKeyHandler
from research.quantizer.v1_lssc_with_perbit_confidence import (
    binarize_with_perbit_confidence,
)
from research.modulation.v2_empirical_llr import EmpiricalLLR

N, m, Z = 52, 42, 16
FULL_BITS = 1536


# ----------------------------------------------------------------------
# Margin Selection Handler
# ----------------------------------------------------------------------
class MarginSelectionHandler(SecureWiFaKeyHandler):
    def enroll(self, feature_vector_float):
        b_full = self._binarize_full(feature_vector_float).astype(np.uint8)
        projected = np.dot(feature_vector_float, self.M_matrix)
        _, margin = binarize_with_perbit_confidence(projected, self.intervals)
        selection_indices = np.argpartition(-margin, self.feature_length)[
            : self.feature_length
        ]
        selection_indices.sort()
        selection_mask = np.zeros(FULL_BITS, dtype=np.uint8)
        selection_mask[selection_indices] = 1
        b_selected = b_full[selection_indices]
        random_key = np.random.randint(0, 2, size=(1, self.key_length), dtype=int)
        codeword = self.encoder.encode_LDPC(random_key).flatten().astype(np.uint8)
        helper_data = np.logical_xor(b_selected, codeword).astype(np.uint8)
        key_hash = hashlib.sha256(random_key.flatten().tobytes()).digest()
        return helper_data, selection_mask, key_hash


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------
def load_embedding(cache_dir, name, imagenum):
    return np.load(os.path.join(cache_dir, f"{name}_{int(imagenum):04d}.npy"))


def load_pairs(pairs_csv, max_pairs=None):
    rows = []
    with open(pairs_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return rows


def pairs_iter(pairs_csv, cache_dir, max_pairs=None):
    rows = load_pairs(pairs_csv, max_pairs)
    for row in rows:
        try:
            e1 = load_embedding(cache_dir, row["name_enroll"], row["imagenum_enroll"])
            e2 = load_embedding(cache_dir, row["name_verify"], row["imagenum_verify"])
            yield e1, e2, row
        except:
            continue


# ----------------------------------------------------------------------
# Decode helpers
# ----------------------------------------------------------------------
def _try_decode(handler, llr_flat, key_hash):
    llr = llr_flat.reshape(1, N, Z)
    y_pred = handler.sess.run(
        handler.decoder_output, feed_dict={handler.xa: llr}
    ).flatten()
    decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
    return hashlib.sha256(decoded_key.tobytes()).digest() == key_hash


def multi_start_decode(handler, llr_flat, key_hash, K, sigma):
    if _try_decode(handler, llr_flat, key_hash):
        return True
    if K <= 0:
        return False
    llr_clean = llr_flat.reshape(1, N, Z)
    for _ in range(K):
        noise = np.random.normal(0, sigma, size=llr_clean.shape).astype(np.float32)
        llr_noisy = llr_clean + noise
        y_pred = handler.sess.run(
            handler.decoder_output, feed_dict={handler.xa: llr_noisy}
        ).flatten()
        decoded_key = (y_pred > 0).astype(int)[: handler.key_length]
        if hashlib.sha256(decoded_key.tobytes()).digest() == key_hash:
            return True
    return False


# ----------------------------------------------------------------------
# Neural Correction loader (singleton để tránh load lại mỗi lần)
# ----------------------------------------------------------------------
_neural_cache = {}


def get_neural_model(model_path, num_layers, hidden_units, activation):
    """Trả về session và các tensor cần thiết cho Neural Correction."""
    key = (model_path, num_layers, hidden_units, activation)
    if key in _neural_cache:
        return _neural_cache[key]

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():
        x_margin = tf.placeholder(tf.float32, [1, 832], name="margin")
        x_llr = tf.placeholder(tf.float32, [1, 832], name="llr_in")
        features = tf.stack([x_margin, x_llr], axis=-1)
        flat = tf.reshape(features, [1, -1])
        h = flat
        for i in range(num_layers):
            act = tf.nn.relu
            if activation == "leaky_relu":
                act = tf.nn.leaky_relu
            elif activation == "swish":
                act = lambda x: x * tf.nn.sigmoid(x)
            h = tf.layers.dense(h, hidden_units, activation=act, name=f"fc{i}")
        out = tf.layers.dense(h, 1, activation=None, name="output")
        pred_llr = tf.reshape(out, [1, -1])
        saver = tf.train.Saver()
        sess = tf.Session(graph=g)
        saver.restore(sess, model_path)
        result = (sess, pred_llr, x_margin, x_llr)
        _neural_cache[key] = result
        return result


# ----------------------------------------------------------------------
# Main benchmark for one config
# ----------------------------------------------------------------------
def run_benchmark(args):
    data_dir = os.path.join(args.project_root, "wifakey_module", "data")
    weights_path = os.path.join(data_dir, "Weights_Var_MS")
    biases_path = os.path.join(data_dir, "Biases_Var_MS")

    if args.selection == "margin":
        handler = MarginSelectionHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )
    else:
        handler = SecureWiFaKeyHandler(
            data_path=data_dir, weights_path=weights_path, biases_path=biases_path
        )

    # Oracle‑LDA
    if args.oracle_lda:
        new_M = np.load(args.oracle_lda)
        if new_M.shape == handler.M_matrix.shape:
            handler.M_matrix = new_M
            print(f"[LDA] Đã thay M_matrix bằng {args.oracle_lda}")
        else:
            print(f"[WARN] M_matrix LDA không khớp shape, bỏ qua.")

    # Empirical LLR
    emp_mod = None
    if args.llr == "empirical":
        emp_mod = EmpiricalLLR(lookup_path=args.lookup)

    # Neural Correction (nếu có)
    neural_sess = None
    neural_pred = None
    neural_margin_ph = None
    neural_llr_ph = None
    if args.neural_model:
        neural_sess, neural_pred, neural_margin_ph, neural_llr_ph = get_neural_model(
            args.neural_model,
            args.neural_layers,
            args.neural_units,
            args.neural_activation,
        )

    pairs_dir = os.path.join(
        args.project_root, "datasets", "processed", args.dataset, "pairs"
    )
    cache_dir = os.path.join(
        args.project_root, "datasets", "processed", args.dataset, "embeddings_cache"
    )
    genuine_csv = os.path.join(pairs_dir, f"{args.tier}_genuine.csv")
    impostor_csv = os.path.join(pairs_dir, f"{args.tier}_impostor.csv")

    results = {"genuine": [], "impostor": []}
    gen_succ, gen_total = 0, 0
    imp_succ, imp_total = 0, 0

    # Genuine
    for feat_enroll, feat_verify, row in pairs_iter(
        genuine_csv, cache_dir, args.max_pairs
    ):
        gen_total += 1
        helper, sel_mask, key_hash = handler.enroll(feat_enroll)
        b_full_v = handler._binarize_full(feat_verify).astype(np.uint8)
        idx = np.where(sel_mask == 1)[0]
        b_sel = b_full_v[idx]
        noisy = np.logical_xor(b_sel, helper).astype(np.uint8)

        # Tính margin (cần cho cả empirical và neural)
        _, margin_v = binarize_with_perbit_confidence(
            np.dot(feat_verify, handler.M_matrix), handler.intervals
        )
        margin_sel = margin_v[idx]

        # Tính LLR cơ bản
        if emp_mod is not None:
            llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
        else:
            llr = 2 * noisy.astype(np.float32) - 1

        # Neural Correction
        if neural_sess is not None:
            margin_input = margin_sel.reshape(1, -1)
            llr_input = llr.reshape(1, -1)
            llr = neural_sess.run(
                neural_pred,
                feed_dict={neural_margin_ph: margin_input, neural_llr_ph: llr_input},
            ).flatten()

        # Giải mã (có thể multi‑start)
        ok = multi_start_decode(handler, llr, key_hash, args.multi_K, args.multi_sigma)
        if ok:
            gen_succ += 1
        pair_id = f"{row['name_enroll']}_{row['imagenum_enroll']}_{row['name_verify']}_{row['imagenum_verify']}"
        results["genuine"].append({"pair_id": pair_id, "success": bool(ok)})

    # Impostor
    if os.path.exists(impostor_csv):
        for feat_enroll, feat_verify, row in pairs_iter(
            impostor_csv, cache_dir, args.max_pairs
        ):
            imp_total += 1
            helper, sel_mask, key_hash = handler.enroll(feat_enroll)
            b_full_v = handler._binarize_full(feat_verify).astype(np.uint8)
            idx = np.where(sel_mask == 1)[0]
            b_sel = b_full_v[idx]
            noisy = np.logical_xor(b_sel, helper).astype(np.uint8)

            _, margin_v = binarize_with_perbit_confidence(
                np.dot(feat_verify, handler.M_matrix), handler.intervals
            )
            margin_sel = margin_v[idx]

            if emp_mod is not None:
                llr = emp_mod.modulate(noisy, context={"margin": margin_sel}).flatten()
            else:
                llr = 2 * noisy.astype(np.float32) - 1

            if neural_sess is not None:
                margin_input = margin_sel.reshape(1, -1)
                llr_input = llr.reshape(1, -1)
                llr = neural_sess.run(
                    neural_pred,
                    feed_dict={
                        neural_margin_ph: margin_input,
                        neural_llr_ph: llr_input,
                    },
                ).flatten()

            ok = multi_start_decode(
                handler, llr, key_hash, args.multi_K, args.multi_sigma
            )
            if ok:
                imp_succ += 1
            pair_id = f"{row['name_enroll']}_{row['imagenum_enroll']}_{row['name_verify']}_{row['imagenum_verify']}"
            results["impostor"].append({"pair_id": pair_id, "success": bool(ok)})

    output_data = {
        "label": args.label,
        "dataset": args.dataset,
        "tier": args.tier,
        "selection": args.selection,
        "llr": args.llr,
        "multi_K": args.multi_K,
        "multi_sigma": args.multi_sigma,
        "oracle_lda": bool(args.oracle_lda),
        "neural_model": bool(args.neural_model),
        "genuine_success": gen_succ,
        "genuine_total": gen_total,
        "impostor_success": imp_succ,
        "impostor_total": imp_total,
        "GMR": gen_succ / gen_total if gen_total else 0,
        "FAR": imp_succ / imp_total if imp_total else 0,
        "results": {},
    }
    for entry in results["genuine"]:
        output_data["results"][entry["pair_id"]] = {
            "is_genuine": True,
            "success": entry["success"],
        }
    for entry in results["impostor"]:
        output_data["results"][entry["pair_id"]] = {
            "is_genuine": False,
            "success": entry["success"],
        }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"GMR: {gen_succ}/{gen_total} ({100*gen_succ/gen_total:.2f}%)")
    print(f"FAR: {imp_succ}/{imp_total} ({100*imp_succ/imp_total:.4f}%)")
    handler.sess.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", choices=["random", "margin"], default="random")
    ap.add_argument("--llr", choices=["bpsk", "empirical"], default="bpsk")
    ap.add_argument("--multi-K", type=int, default=0)
    ap.add_argument("--multi-sigma", type=float, default=0.2)
    ap.add_argument("--oracle-lda", default=None)
    ap.add_argument("--neural-model", default=None)
    ap.add_argument("--neural-layers", type=int, default=3)
    ap.add_argument("--neural-units", type=int, default=128)
    ap.add_argument("--neural-activation", default="relu")
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--dataset", default="labeled_faces_in_the_wild")
    ap.add_argument("--tier", default="tune")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="config")
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    run_benchmark(args)


if __name__ == "__main__":
    main()
