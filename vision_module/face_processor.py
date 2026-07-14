import cv2
import os
import numpy as np
from skimage import transform as trans
from insightface.app import FaceAnalysis
import onnxruntime as ort


class FaceProcessor:
    def __init__(
        self,
        det_model: str = "buffalo_l",
        anti_spoofing_filename: str = "anti-spoofing.onnx",
        ctx_id: int = 0,  # 0 = GPU, -1 = CPU
        det_size=(640, 640),
        confidence_threshold: float = 0.6,
        liveness_threshold: float = 0.8,
        bbox_expansion_factor: float = 1.5,
    ):
        # ================= PATH MODEL =================
        base_dir = os.path.dirname(os.path.abspath(__file__))
        anti_spoofing_path = os.path.join(base_dir, anti_spoofing_filename)

        if not os.path.exists(anti_spoofing_path):
            raise FileNotFoundError(f"Không tìm thấy model: {anti_spoofing_path}")

        print(f"🔎 Loading anti-spoof model từ: {anti_spoofing_path}")
        # ---------------- FACE DETECTOR ----------------
        self.app = FaceAnalysis(
            name=det_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # ---------------- LIVENESS MODEL ----------------
        self.liveness_session = ort.InferenceSession(
            anti_spoofing_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        # Tự động đọc input size từ ONNX
        input_shape = self.liveness_session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        print(f"🔎 Anti-spoof input size: {self.input_width}x{self.input_height}")

        # Số class output
        output_shape = self.liveness_session.get_outputs()[0].shape
        self.num_classes = output_shape[1]
        print(f"🔎 Anti-spoof output classes: {self.num_classes}")

        self.confidence_threshold = confidence_threshold
        self.liveness_threshold = liveness_threshold
        self.bbox_expansion_factor = bbox_expansion_factor
        # SuriAI compares (real_logit - spoof_logit) against logit(threshold_prob).
        # threshold=0.5 -> logit threshold 0.0.
        p = float(np.clip(self.liveness_threshold, 1e-6, 1.0 - 1e-6))
        self.liveness_logit_threshold = float(np.log(p / (1.0 - p))) #note

        # Landmark chuẩn InsightFace (112x112)
        self.REFERENCE_POINTS = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )

        print("✅ FaceProcessor đã khởi tạo xong.")

    # =====================================================
    # LIVENESS CHECK
    # =====================================================
    def _crop_square_reflect(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox.astype(int)
        w = x2 - x1
        h = y2 - y1
        max_dim = max(w, h)
        center_x = x1 + w / 2.0
        center_y = y1 + h / 2.0
        crop_size = int(max_dim * self.bbox_expansion_factor)

        x = int(center_x - crop_size / 2.0)
        y = int(center_y - crop_size / 2.0)

        original_height, original_width = image.shape[:2]
        crop_x1 = max(0, x)
        crop_y1 = max(0, y)
        crop_x2 = min(original_width, x + crop_size)
        crop_y2 = min(original_height, y + crop_size)

        top_pad = int(max(0, -y))
        left_pad = int(max(0, -x))
        bottom_pad = int(max(0, (y + crop_size) - original_height))
        right_pad = int(max(0, (x + crop_size) - original_width))

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return np.zeros((crop_size, crop_size, 3), dtype=image.dtype)

        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2, :]
        return cv2.copyMakeBorder(
            cropped,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            cv2.BORDER_REFLECT_101,
        )

    def _preprocess_liveness(self, image_rgb: np.ndarray) -> np.ndarray:
        target_size = int(self.input_width)
        old_h, old_w = image_rgb.shape[:2]
        ratio = float(target_size) / max(old_h, old_w)
        scaled_h = int(old_h * ratio)
        scaled_w = int(old_w * ratio)

        interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(image_rgb, (scaled_w, scaled_h), interpolation=interpolation)

        delta_w = target_size - scaled_w
        delta_h = target_size - scaled_h
        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_REFLECT_101
        )
        chw = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(chw, axis=0)

    def _check_liveness(self, image: np.ndarray, face) -> float:
        bbox = face.bbox.astype(int)
        face_img = self._crop_square_reflect(image, bbox)

        if face_img.size == 0:
            return 0.0

        # SuriAI model is trained with RGB crops and letterbox preprocessing.
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = self._preprocess_liveness(face_img_rgb)

        # Inference
        input_name = self.liveness_session.get_inputs()[0].name
        outputs = self.liveness_session.run(None, {input_name: img})
        logits = outputs[0][0]

        # SuriAI decision logic: real_logit - spoof_logit.
        real_logit = float(logits[0])
        spoof_logit = float(logits[1])
        return real_logit - spoof_logit

    # =====================================================
    # ALIGN FACE (CHO EMBEDDING)
    # =====================================================
    def _align_face(self, img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        src = landmarks.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(src, self.REFERENCE_POINTS)
        M = tform.params[0:2, :]
        return cv2.warpAffine(
            img,
            M,
            (112, 112),
            flags=cv2.INTER_LINEAR,
            borderValue=0.0,
        )

    # =====================================================
    # MAIN PROCESS
    # =====================================================
    def process(self, image: np.ndarray):
        if image is None:
            return None, "no_image"

        faces = self.app.get(image)
        if not faces:
            return None, "no_face"

        valid_faces = [
            f for f in faces if f.det_score >= self.confidence_threshold
        ]
        if not valid_faces:
            return None, "low_confidence"

        # Anti-spoof từng mặt: có bất kỳ mặt spoof trong khung => từ chối ngay.
        for f in valid_faces:
            score = self._check_liveness(image, f)
            if score < self.liveness_logit_threshold:
                print(f"⚠️ Spoof detected! Logit diff: {score:.3f}")
                return None, "spoof_detected"

        # Tất cả mặt hợp lệ đều pass liveness → chọn mặt lớn nhất để align.
        best_face = max(
            valid_faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        # Lấy landmarks
        landmarks = getattr(
            best_face,
            "kps",
            getattr(best_face, "landmark_2d_5", None),
        )

        if landmarks is None:
            return None, "no_landmarks"

        aligned_bgr = self._align_face(image, landmarks)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        return aligned_rgb, "success"