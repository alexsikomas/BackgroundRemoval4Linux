import cv2 as cv
import onnxruntime as ort
import numpy as np

class RVMInference:
    def __init__(self, session: ort.InferenceSession) -> None:
        self.session = session
        self.input_map = {}
        for node in self.session.get_inputs():
            if 'float16' in node.type:
                self.input_map[node.name] = np.float16
            else:
                self.input_map[node.name] = np.float32
        dtype_r = self.input_map.get('r1i', np.float32)

        self.rec = {
            'r1i': np.zeros([1, 1, 1, 1], dtype=dtype_r),
            'r2i': np.zeros([1, 1, 1, 1], dtype=dtype_r),
            'r3i': np.zeros([1, 1, 1, 1], dtype=dtype_r),
            'r4i': np.zeros([1, 1, 1, 1], dtype=dtype_r)
        }

    def process(self, src: np.ndarray, downsample_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        dtype_src = self.input_map.get('src', np.float32)
        dtype_ds = self.input_map.get('downsample_ratio', np.float32)

        inputs = {
            'src': src.astype(dtype_src),
            **self.rec,
            'downsample_ratio': np.array([downsample_ratio], dtype=dtype_ds)
        }

        fgr, alpha, r1o, r2o, r3o, r4o = self.session.run(None, inputs)

        dtype_r = self.input_map.get('r1i', np.float32)
        self.rec['r1i'] = np.array(r1o, dtype=dtype_r)
        self.rec['r2i'] = np.array(r2o, dtype=dtype_r)
        self.rec['r3i'] = np.array(r3o, dtype=dtype_r)
        self.rec['r4i'] = np.array(r4o, dtype=dtype_r)

        # float32 for opencv
        fgr = np.array(fgr, dtype=np.float32)
        alpha = np.array(alpha, dtype=np.float32)

        fgr = cv.convertScaleAbs(fgr, alpha=255.0)
        alpha = cv.convertScaleAbs(alpha, alpha=255.0)

        fgr = np.ascontiguousarray(np.transpose(fgr[0], (1, 2, 0)))
        alpha = np.ascontiguousarray(np.transpose(alpha[0], (1, 2, 0)))

        return fgr, alpha
