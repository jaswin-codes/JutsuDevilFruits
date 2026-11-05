# sign_model.py - real Naruto hand signs from a trained YOLOX-Nano detector.
# Model and pre/post-processing adapted from Kazuhito00/NARUTO-HandSignDetection
# (MIT licence, Copyright (c) 2020 KazuhitoTakahashi; see model/LICENSE).
import os

import cv2
import numpy as np
import onnxruntime

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "yolox_nano_with_post.onnx")
INPUT_SIZE = 416
# Model class id -> sign name; the model's "Unknown" and "Mizunoe" classes count as no sign
LABELS = ["Ne", "Ushi", "Tora", "U", "Tatsu", "Mi", "Uma", "Hitsuji",
          "Saru", "Tori", "Inu", "I", "Gassho", None, None]


def best_sign(dets, ratio, score_th):
    """dets rows: [batch, class, score, x1, y1, x2, y2] in model pixels.
    Returns (sign, (x1, y1, x2, y2) in frame pixels) for the top detection, or (None, None)."""
    if dets is None or len(dets) == 0:
        return None, None
    _, class_id, score, *box = dets[np.argmax(dets[:, 2])]
    class_id = int(class_id)
    if score < score_th or not 0 <= class_id < len(LABELS) or LABELS[class_id] is None:
        return None, None
    return LABELS[class_id], tuple(int(v / ratio) for v in box)


class SignModel:
    def __init__(self, path=MODEL_PATH, score_th=0.6):
        self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.score_th = score_th

    def detect(self, frame):
        """frame: BGR image. Returns (sign name or None, bounding box or None)."""
        ratio = min(INPUT_SIZE / frame.shape[0], INPUT_SIZE / frame.shape[1])
        resized = cv2.resize(frame, (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)   # grey pad, as trained
        padded[:resized.shape[0], :resized.shape[1]] = resized
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        dets = self.session.run(None, {self.input_name: blob})[0]
        return best_sign(dets, ratio, self.score_th)
