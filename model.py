import cv2 as cv
import onnxruntime as ort
import numpy as np

class RVMInference:
    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session
        self.rec = {
            'r1i': np.zeros([1, 1, 1, 1], dtype=np.float32),
            'r2i': np.zeros([1, 1, 1, 1], dtype=np.float32),
            'r3i': np.zeros([1, 1, 1, 1], dtype=np.float32),
            'r4i': np.zeros([1, 1, 1, 1], dtype=np.float32)
        }

    def process(self, src: np.ndarray, downsample_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        inputs = {
            'src': src.astype(np.float32),
            **self.rec,
            'downsample_ratio': np.array([downsample_ratio], dtype=np.float32)
        }

        fgr, alpha, r1o, r2o, r3o, r4o = self.session.run(None, inputs)

        self.rec['r1i'] = np.array(r1o, dtype=np.float32)
        self.rec['r2i'] = np.array(r2o, dtype=np.float32)
        self.rec['r3i'] = np.array(r3o, dtype=np.float32)
        self.rec['r4i'] = np.array(r4o, dtype=np.float32)
        fgr = np.array(fgr, dtype=np.float32)
        alpha = np.array(alpha, dtype=np.float32)

        fgr = cv.convertScaleAbs(fgr, alpha=255.0)
        alpha = cv.convertScaleAbs(alpha, alpha=255.0)

        fgr = np.ascontiguousarray(np.transpose(fgr[0], (1, 2, 0)))
        alpha = np.ascontiguousarray(np.transpose(alpha[0], (1, 2, 0)))

        return fgr, alpha
