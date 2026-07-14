import cv2
import numpy as np
from skimage import transform as trans

# Standard InsightFace 112x112 reference landmarks
_REF = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(img_bgr: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Warp face to 112x112 RGB using 5-point similarity transform."""
    tform = trans.SimilarityTransform()
    tform.estimate(keypoints, _REF)
    M = tform.params[:2]
    aligned = cv2.warpAffine(img_bgr, M, (112, 112), flags=cv2.INTER_LINEAR, borderValue=0)
    return cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
