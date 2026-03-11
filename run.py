import argparse
import sys
import threading
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pyvirtualcam
from pathlib import Path
from typing import Optional, Tuple
from pyvirtualcam import PixelFormat

DEFAULT_MODEL_PATH = str(
    Path.home() / "rvm_models" / "rvm_resnet50_fp16.torchscript"
)
DEFAULT_WEBCAM = 0
DEFAULT_VCAM = "/dev/video10"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 60
DEFAULT_DOWNSAMPLE = 0.25

class GaussianBlurGPU:
    def __init__(
        self,
        kernel_size: int,
        sigma: float,
        device: torch.device,
        dtype: torch.dtype,
        fast_downscale: int = 4,
    ):
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.orig_kernel_size = kernel_size
        self.fast_downscale = fast_downscale

        # When downscaling, shrink the kernel proportionally
        if fast_downscale > 1:
            eff_k = max(kernel_size // fast_downscale, 3)
            if eff_k % 2 == 0:
                eff_k += 1
            eff_sigma = sigma / fast_downscale
        else:
            eff_k = kernel_size
            eff_sigma = sigma

        self.eff_k = eff_k
        self.padding = eff_k // 2

        x = torch.arange(eff_k, dtype=torch.float32) - (eff_k - 1) / 2.0
        gauss = torch.exp(-0.5 * (x / eff_sigma) ** 2)
        gauss /= gauss.sum()

        k_h = gauss.view(1, 1, -1, 1).repeat(3, 1, 1, 1)   # vertical
        k_w = gauss.view(1, 1, 1, -1).repeat(3, 1, 1, 1)   # horizontal

        self.kernel_h = k_h.to(device=device, dtype=dtype)
        self.kernel_w = k_w.to(device=device, dtype=dtype)

    @torch.inference_mode()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Blur a [1,3,H,W] tensor."""
        _, _, H, W = img.shape

        if self.fast_downscale > 1:
            small = F.interpolate(
                img,
                scale_factor=1.0 / self.fast_downscale,
                mode="bilinear",
                align_corners=False,
            )
            out = F.conv2d(small, self.kernel_h,
                           padding=(self.padding, 0), groups=3)
            out = F.conv2d(out, self.kernel_w,
                           padding=(0, self.padding), groups=3)
            out = F.interpolate(out, size=(H, W),
                                mode="bilinear", align_corners=False)
        else:
            out = F.conv2d(img, self.kernel_h,
                           padding=(self.padding, 0), groups=3)
            out = F.conv2d(out, self.kernel_w,
                           padding=(0, self.padding), groups=3)
        return out


def make_green_bg(
    h: int, w: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    bg = torch.zeros(1, 3, h, w, device=device, dtype=dtype)
    bg[0, 1, :, :] = 177.0 / 255.0
    return bg


def make_solid_bg(
    h: int, w: int, device: torch.device, dtype: torch.dtype,
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    bg = torch.zeros(1, 3, h, w, device=device, dtype=dtype)
    for c in range(3):
        bg[0, c, :, :] = color[c]
    return bg


def load_image_bg(
    path: str, h: int, w: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load background image: {path}")
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bg = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return bg.to(device=device, dtype=dtype).div_(255.0)


class ThreadedWebcamCapture:
    """
    A background thread calls cap.read() in a tight loop, storing
    only the *latest* frame.  The main thread never blocks on V4L2
    dequeue and always gets the most recent image — this is the
    single biggest latency win when the processing loop is slower
    than the camera frame rate.
    """

    def __init__(self, device: int, width: int, height: int, fps: int):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam /dev/video{device}")

        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(
            f"[Webcam] /dev/video{device}: "
            f"{self.width}x{self.height} @ {actual_fps:.0f}fps"
        )

        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._running = True

        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
                self._new_frame.set()

    def read(self) -> Optional[np.ndarray]:
        """Return the latest captured frame (non-blocking)."""
        with self._lock:
            return self._frame

    def wait_and_read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Block until a new frame arrives, then return it."""
        self._new_frame.wait(timeout=timeout)
        self._new_frame.clear()
        return self.read()

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()


class FramePreprocessor:
    """
    BGR uint8 numpy → [1,3,H,W] normalised GPU tensor.
    Uses pinned memory + non-blocking upload.
    cv2.cvtColor is used for BGR→RGB (faster than numpy slice reversal
    and always returns a contiguous array).
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self._pinned: Optional[torch.Tensor] = None

    def __call__(self, frame_bgr: np.ndarray) -> torch.Tensor:
        h, w = frame_bgr.shape[:2]

        if self._pinned is None or self._pinned.shape[:2] != (h, w):
            self._pinned = torch.empty(
                h, w, 3, dtype=torch.uint8, pin_memory=True
            )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        np.copyto(self._pinned.numpy(), frame_rgb)

        gpu = self._pinned.to(self.device, non_blocking=True)
        gpu = gpu.permute(2, 0, 1).unsqueeze(0)   # [1,3,H,W]
        gpu = gpu.to(self.dtype).div_(255.0)
        return gpu


class RVMEngine:
    """
    RVM TorchScript wrapper.  All work on the default CUDA stream
    (no inter-stream races).
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        downsample_ratio: float,
    ):
        self.device = device
        self.dtype = dtype
        self.downsample_ratio = downsample_ratio

        print(f"[RVM] Loading: {model_path}")
        print(f"[RVM] Device={device}  Dtype={dtype}  DS={downsample_ratio}")

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        try:
            self.model = torch.jit.freeze(self.model)
            print("[RVM] Model frozen (graph optimised)")
        except Exception as e:
            print(f"[RVM] Could not freeze model (non-critical): {e}")

        self.rec = [None] * 4
        self._warmup()
        print("[RVM] Ready.")

    def _warmup(self, n: int = 5) -> None:
        dummy = torch.randn(
            1, 3, 720, 1280, device=self.device, dtype=self.dtype
        )
        rec = [None] * 4
        with torch.inference_mode():
            for _ in range(n):
                _fgr, _pha, *rec = self.model(
                    dummy, *rec, self.downsample_ratio
                )
        self.rec = [None] * 4
        torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def __call__(
        self, src: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fgr, pha, *self.rec = self.model(
            src, *self.rec, self.downsample_ratio
        )
        return fgr, pha

    def reset(self) -> None:
        self.rec = [None] * 4


class Compositor:
    def __init__(self, height: int, width: int):
        self._out = np.empty((height, width, 3), dtype=np.uint8)

    @torch.inference_mode()
    def __call__(
        self,
        src: torch.Tensor,
        pha: torch.Tensor,
        bg: torch.Tensor,
    ) -> np.ndarray:
        comp = src * pha + bg * (1.0 - pha)
        comp = (
            comp.squeeze(0)
            .mul(255.0)
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
        )
        torch.cuda.synchronize()
        np.copyto(self._out, comp.cpu().numpy())
        return self._out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RVM ResNet50 real-time webcam matting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Blur background (default):
  python run.py

  # Green screen:
  python run.py --bg-mode green

  # Custom background image:
  python run.py --bg-mode image --bg-image ~/wallpaper.jpg

  # Solid color (white):
  python run.py --bg-mode color --bg-color 255,255,255

  # 1080p (needs RTX 3060+):
  python run.py --width 1920 --height 1080 --downsample 0.2
""",
    )

    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--webcam", type=int, default=DEFAULT_WEBCAM)
    parser.add_argument("--vcam", default=DEFAULT_VCAM)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--downsample", type=float, default=DEFAULT_DOWNSAMPLE)

    bg = parser.add_argument_group("Background")
    bg.add_argument(
        "--bg-mode",
        default="blur",                         # ← CHANGED from "green"
        choices=["green", "image", "color", "blur"],
    )
    bg.add_argument("--bg-image", default=None)
    bg.add_argument(
        "--bg-color", default="0,177,0", help="R,G,B 0-255"
    )
    bg.add_argument(
        "--blur-radius",
        type=int,
        default=51,
        help="Gaussian blur kernel size (odd, blur mode)",
    )
    bg.add_argument(
        "--blur-sigma",
        type=float,
        default=0.0,
        help="Blur sigma (0 = auto from kernel size)",
    )
    bg.add_argument(
        "--blur-downscale",
        type=int,
        default=4,
        help="Downscale factor for fast blur (1 = full-res, 4 = 4× faster)",
    )

    args = parser.parse_args()

    if not Path(args.model).is_file():
        print(f"ERROR: Model not found at {args.model}")
        print("Run setup.sh or download from:")
        print("https://github.com/PeterL1n/RobustVideoMatting/releases")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)

    if major >= 7:
        dtype = torch.float16
        print(f"[GPU] {gpu_name} (sm_{major}{minor}) → FP16")
    else:
        dtype = torch.float32
        print(f"[GPU] {gpu_name} (sm_{major}{minor}) → FP32 (pre-Turing)")
        print("      Consider rvm_resnet50_fp32.torchscript for this GPU.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    webcam = ThreadedWebcamCapture(
        args.webcam, args.width, args.height, args.fps
    )
    H, W = webcam.height, webcam.width

    engine = RVMEngine(args.model, device, dtype, args.downsample)

    preprocess = FramePreprocessor(device, dtype)

    compositor = Compositor(H, W)

    gpu_blur: Optional[GaussianBlurGPU] = None
    static_bg: Optional[torch.Tensor] = None

    if args.bg_mode == "green":
        static_bg = make_green_bg(H, W, device, dtype)
    elif args.bg_mode == "image":
        if not args.bg_image:
            print("ERROR: --bg-image required with --bg-mode image")
            sys.exit(1)
        static_bg = load_image_bg(args.bg_image, H, W, device, dtype)
    elif args.bg_mode == "color":
        r, g, b = [
            int(x.strip()) / 255.0 for x in args.bg_color.split(",")
        ]
        static_bg = make_solid_bg(H, W, device, dtype, (r, g, b))
    elif args.bg_mode == "blur":
        sigma = (
            args.blur_sigma
            if args.blur_sigma > 0
            else 0.3 * ((args.blur_radius - 1) * 0.5 - 1) + 0.8
        )
        gpu_blur = GaussianBlurGPU(
            args.blur_radius,
            sigma,
            device,
            dtype,
            fast_downscale=args.blur_downscale,
        )
        print(
            f"[Blur] GPU Gaussian blur: kernel={args.blur_radius}, "
            f"sigma={sigma:.2f}, downscale={args.blur_downscale}×"
        )

    print(f"[vCam] Opening {args.vcam} ({W}x{H} @ {args.fps}fps)")
    vcam = pyvirtualcam.Camera(
        width=W,
        height=H,
        fps=args.fps,
        fmt=PixelFormat.RGB,
        device=args.vcam,
    )
    print(f"[vCam] Active: {vcam.device}")

    print(f"\n{'=' * 60}")
    print(f"  LIVE — Ctrl+C to stop")
    print(f"  Mode: {args.bg_mode} | {W}x{H} @ {args.fps}fps")
    print(f"{'=' * 60}\n")

    try:
        while True:
            frame_bgr = webcam.wait_and_read()
            if frame_bgr is None:
                continue

            src = preprocess(frame_bgr)

            if args.bg_mode == "blur":
                bg = gpu_blur(src)
            else:
                bg = static_bg

            fgr, pha = engine(src)

            frame_rgb = compositor(src, pha, bg)

            vcam.send(frame_rgb)
    finally:
        webcam.release()
        vcam.close()
        if args.preview:
            cv2.destroyAllWindows()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
