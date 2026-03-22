import argparse
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyvirtualcam
import torch
import torch.nn.functional as F
from pyvirtualcam import PixelFormat

# Defaults
DEFAULT_MODEL_PATH = str(
    Path.home() / "rvm_models" / "rvm_resnet50_fp16.torchscript"
)
DEFAULT_WEBCAM = 0
DEFAULT_VCAM = "/dev/video10"
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 60
DEFAULT_DOWNSAMPLE = 0.25
DEFAULT_BG_COLOR = (0, 177, 0)
DEFAULT_BG_COLOR_STR = ",".join(map(str, DEFAULT_BG_COLOR))

PREVIEW_MAX_FPS = 30
PREVIEW_INTERVAL_S = 1.0 / PREVIEW_MAX_FPS
BG_MODES = ("blur", "green", "image", "color")

FALLBACK_RESOLUTIONS: List[Tuple[int, int, List[int]]] = [
    (1920, 1080, [60, 30]),
    (1280, 720, [60, 30]),
    (640, 480, [30]),
]


# Data
@dataclass
class PipelineConfig:
    model_path: str = DEFAULT_MODEL_PATH
    webcam: int = DEFAULT_WEBCAM
    vcam: str = DEFAULT_VCAM
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    downsample: float = DEFAULT_DOWNSAMPLE
    bg_mode: str = "blur"
    bg_image: Optional[str] = None
    bg_color: Tuple[int, int, int] = DEFAULT_BG_COLOR
    blur_radius: int = 51
    blur_sigma: float = 0.0
    blur_downscale: int = 4


@dataclass
class PipelineRuntime:
    device: torch.device
    dtype: torch.dtype
    gpu_name: str
    webcam: "ThreadedWebcamCapture"
    engine: "RVMEngine"
    preprocess: "FramePreprocessor"
    compositor: "Compositor"
    static_bg: Optional[torch.Tensor]
    gpu_blur: Optional["GaussianBlurGPU"]
    height: int
    width: int


# helpers
def parse_bg_color(value: str) -> Tuple[int, int, int]:
    try:
        color = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError("--bg-color must be in R,G,B format") from exc
    if len(color) != 3 or any(not 0 <= c <= 255 for c in color):
        raise ValueError("--bg-color must contain exactly 3 values in 0-255")
    return color


def resolve_blur_sigma(radius: int, sigma: float) -> float:
    return sigma if sigma > 0 else 0.3 * ((radius - 1) * 0.5 - 1) + 0.8


def normalize_rgb(color: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return tuple(c / 255.0 for c in color)


def bg_mode_from_id(idx: int) -> str:
    return BG_MODES[idx] if 0 <= idx < len(BG_MODES) else BG_MODES[0]


def bg_mode_to_id(mode: str) -> int:
    return BG_MODES.index(mode) if mode in BG_MODES else 0


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        model_path=args.model,
        webcam=args.webcam,
        vcam=args.vcam,
        width=args.width,
        height=args.height,
        fps=args.fps,
        downsample=args.downsample,
        bg_mode=args.bg_mode,
        bg_image=args.bg_image,
        bg_color=parse_bg_color(args.bg_color),
        blur_radius=args.blur_radius,
        blur_sigma=args.blur_sigma,
        blur_downscale=args.blur_downscale,
    )


def emit(callback: Optional[Callable[[str], None]], message: str) -> None:
    if callback is not None:
        callback(message)


def close_runtime(runtime: Optional[PipelineRuntime]) -> None:
    if runtime is None:
        return
    try:
        runtime.webcam.release()
    except Exception:
        pass


def current_background(
    src: torch.Tensor,
    static_bg: Optional[torch.Tensor],
    gpu_blur: Optional["GaussianBlurGPU"],
) -> torch.Tensor:
    return gpu_blur(src) if gpu_blur is not None else static_bg


def rebuild_background(
    cfg: PipelineConfig,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    **changes: Any,
) -> Tuple[Optional[torch.Tensor], Optional["GaussianBlurGPU"]]:
    old = {key: getattr(cfg, key) for key in changes}
    for key, value in changes.items():
        setattr(cfg, key, value)
    try:
        return build_background(cfg, h, w, device, dtype)
    except Exception:
        for key, value in old.items():
            setattr(cfg, key, value)
        raise


# gpu helpers
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

        if fast_downscale > 1:
            eff_k = max(kernel_size // fast_downscale, 3)
            if eff_k % 2 == 0:
                eff_k += 1
            eff_sigma = sigma / fast_downscale
        else:
            eff_k = kernel_size
            eff_sigma = sigma

        self.padding = eff_k // 2

        x = torch.arange(eff_k, dtype=torch.float32) - (eff_k - 1) / 2.0
        gauss = torch.exp(-0.5 * (x / eff_sigma) ** 2)
        gauss /= gauss.sum()

        self.kernel_h = (
            gauss.view(1, 1, -1, 1).repeat(3, 1, 1, 1).to(device=device, dtype=dtype)
        )
        self.kernel_w = (
            gauss.view(1, 1, 1, -1).repeat(3, 1, 1, 1).to(device=device, dtype=dtype)
        )

    @torch.inference_mode()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, _, h, w = img.shape
        if self.fast_downscale > 1:
            img = F.interpolate(
                img,
                scale_factor=1.0 / self.fast_downscale,
                mode="bilinear",
                align_corners=False,
            )
        img = F.conv2d(img, self.kernel_h, padding=(self.padding, 0), groups=3)
        img = F.conv2d(img, self.kernel_w, padding=(0, self.padding), groups=3)
        if self.fast_downscale > 1:
            img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
        return img


def make_solid_bg(
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    color: Tuple[float, float, float],
) -> torch.Tensor:
    return torch.tensor(color, device=device, dtype=dtype).view(1, 3, 1, 1).expand(
        1, 3, h, w
    )


def load_image_bg(
    path: str, h: int, w: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load background image: {path}")
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
        .div_(255.0)
    )


# webcam
class ThreadedWebcamCapture:
    def __init__(self, device: int, width: int, height: int, fps: int):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam /dev/video{device}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self._new_frame.set()

    def wait_and_read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        self._new_frame.wait(timeout=timeout)
        with self._lock:
            self._new_frame.clear()
            return self._frame

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2.0)
        self.cap.release()


# preprocessing compositing rvm
class FramePreprocessor:
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype
        self._pinned: Optional[torch.Tensor] = None
        self._pinned_np: Optional[np.ndarray] = None

    def __call__(self, frame_bgr: np.ndarray) -> torch.Tensor:
        h, w = frame_bgr.shape[:2]
        if self._pinned is None or self._pinned.shape[:2] != (h, w):
            self._pinned = torch.empty(h, w, 3, dtype=torch.uint8, pin_memory=True)
            self._pinned_np = self._pinned.numpy()
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB, dst=self._pinned_np)
        return (
            self._pinned.to(self.device, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.dtype)
            .div_(255.0)
        )


class RVMEngine:
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

    def warmup(self, h: int, w: int, n: int = 5) -> None:
        dummy = torch.randn(1, 3, h, w, device=self.device, dtype=self.dtype)
        rec = [None] * 4
        with torch.inference_mode():
            for _ in range(n):
                _fgr, _pha, *rec = self.model(dummy, *rec, self.downsample_ratio)
        self.rec = [None] * 4
        torch.cuda.synchronize(self.device)
        print("[RVM] Warmup complete.")

    @torch.inference_mode()
    def __call__(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fgr, pha, *self.rec = self.model(src, *self.rec, self.downsample_ratio)
        return fgr, pha

    def reset(self) -> None:
        self.rec = [None] * 4


class Compositor:
    def __init__(self, height: int, width: int):
        self._pinned = torch.empty(
            height, width, 3, dtype=torch.uint8, pin_memory=True
        )
        self._out: np.ndarray = self._pinned.numpy()

    @torch.inference_mode()
    def __call__(
        self,
        fgr: torch.Tensor,
        pha: torch.Tensor,
        bg: torch.Tensor,
    ) -> np.ndarray:
        comp = (
            torch.lerp(bg, fgr, pha)
            .squeeze(0)
            .mul_(255.0)
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
        )
        self._pinned.copy_(comp)
        return self._out


class FPSCounter:
    def __init__(self, log_interval: float = 5.0):
        self._interval = log_interval
        self._count = 0
        self._t0 = time.monotonic()

    def tick(self) -> Optional[float]:
        self._count += 1
        elapsed = time.monotonic() - self._t0
        if elapsed < self._interval:
            return None
        fps = self._count / elapsed
        self._count = 0
        self._t0 = time.monotonic()
        return fps


def enumerate_v4l2_devices() -> List[Dict[str, Any]]:
    devices = []
    sysfs = Path("/sys/class/video4linux")
    if not sysfs.exists():
        return devices

    for entry in sorted(sysfs.iterdir()):
        name_file = entry / "name"
        if not name_file.exists():
            continue
        try:
            idx = int(entry.name.replace("video", ""))
        except ValueError:
            continue
        devices.append(
            {
                "index": idx,
                "path": f"/dev/{entry.name}",
                "name": name_file.read_text().strip(),
            }
        )
    return devices


def query_camera_formats(
    device_index: int,
) -> Dict[str, List[Tuple[int, int, List[float]]]]:
    """Use ``v4l2-ctl --list-formats-ext`` to discover every pixel-format,
    resolution, and frame-rate a V4L2 camera advertises.

    Returns
    -------
    dict  –  ``{ "MJPG": [(1920, 1080, [60.0, 30.0]), ...], ... }``
    """
    device_path = f"/dev/video{device_index}"
    result: Dict[str, List[Tuple[int, int, List[float]]]] = {}

    try:
        output = subprocess.check_output(
            ["v4l2-ctl", "--list-formats-ext", "-d", device_path],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return result

    current_format: Optional[str] = None
    current_fps_list: Optional[List[float]] = None

    for line in output.splitlines():
        fmt_match = re.match(r"\s*\[\d+]:\s+'(\w+)'", line)
        if fmt_match:
            current_format = fmt_match.group(1)
            result.setdefault(current_format, [])
            current_fps_list = None
            continue

        # Size line:  Size: Discrete 1920x1080
        size_match = re.match(r"\s*Size:\s+Discrete\s+(\d+)x(\d+)", line)
        if size_match and current_format is not None:
            w, h = int(size_match.group(1)), int(size_match.group(2))
            fps_list: List[float] = []
            result[current_format].append((w, h, fps_list))
            current_fps_list = fps_list
            continue

        # Interval line:  Interval: Discrete 0.033s (30.000 fps)
        fps_match = re.match(
            r"\s*Interval:\s+Discrete\s+[\d.]+s\s+\(([\d.]+)\s+fps\)", line
        )
        if fps_match and current_fps_list is not None:
            current_fps_list.append(float(fps_match.group(1)))

    return result


def get_preferred_resolutions(
    device_index: int,
) -> List[Tuple[int, int, List[int]]]:
    """Return a resolution list sorted largest-first for the best available
    pixel format (MJPG preferred, then any other).

    Each entry is ``(width, height, [fps_int, ...])``.
    Falls back to :data:`FALLBACK_RESOLUTIONS` when the query fails.
    """
    formats = query_camera_formats(device_index)

    # Pick the first format we find in priority order
    chosen_entries: Optional[List[Tuple[int, int, List[float]]]] = None
    for fmt_name in ("MJPG", "YUYV", "NV12", "H264"):
        if fmt_name in formats and formats[fmt_name]:
            chosen_entries = formats[fmt_name]
            break
    if chosen_entries is None:
        # Grab whatever is there
        for entries in formats.values():
            if entries:
                chosen_entries = entries
                break

    if not chosen_entries:
        return list(FALLBACK_RESOLUTIONS)

    chosen_entries.sort(key=lambda e: e[0] * e[1], reverse=True)

    result: List[Tuple[int, int, List[int]]] = []
    for w, h, fps_list in chosen_entries:
        fps_ints = sorted(
            {int(round(f)) for f in fps_list}, reverse=True
        )
        if not fps_ints:
            fps_ints = [30]
        result.append((w, h, fps_ints))
    return result


def print_camera_formats(device_index: int) -> None:
    """Pretty-print every format/resolution/fps a camera supports."""
    formats = query_camera_formats(device_index)
    if not formats:
        print(f"  (no formats reported for /dev/video{device_index})")
        return
    for fmt, entries in formats.items():
        print(f"  [{fmt}]")
        for w, h, fps_list in entries:
            fps_str = ", ".join(f"{f:.0f}" for f in fps_list)
            print(f"    {w}x{h}  @  {fps_str} fps")


def setup_gpu() -> Tuple[torch.device, torch.dtype, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    major, _ = torch.cuda.get_device_capability(device)
    dtype = torch.float16 if major >= 7 else torch.float32
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    return device, dtype, gpu_name


def build_background(
    cfg: PipelineConfig,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[torch.Tensor], Optional[GaussianBlurGPU]]:
    if cfg.bg_mode == "blur":
        return None, GaussianBlurGPU(
            cfg.blur_radius,
            resolve_blur_sigma(cfg.blur_radius, cfg.blur_sigma),
            device,
            dtype,
            fast_downscale=cfg.blur_downscale,
        )
    if cfg.bg_mode == "green":
        return (
            make_solid_bg(h, w, device, dtype, normalize_rgb(DEFAULT_BG_COLOR)),
            None,
        )
    if cfg.bg_mode == "image":
        if not cfg.bg_image:
            raise ValueError("--bg-image required with --bg-mode image")
        return load_image_bg(cfg.bg_image, h, w, device, dtype), None
    if cfg.bg_mode == "color":
        return (
            make_solid_bg(h, w, device, dtype, normalize_rgb(cfg.bg_color)),
            None,
        )
    raise ValueError(f"Unsupported background mode: {cfg.bg_mode}")


def init_runtime(
    cfg: PipelineConfig,
    status: Optional[Callable[[str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> PipelineRuntime:
    webcam = None
    try:
        emit(status, "Initialising GPU…")
        device, dtype, gpu_name = setup_gpu()
        emit(
            log,
            f"[GPU] {gpu_name}  "
            f"{'FP16' if dtype == torch.float16 else 'FP32'}",
        )

        emit(status, "Opening webcam…")
        webcam = ThreadedWebcamCapture(cfg.webcam, cfg.width, cfg.height, cfg.fps)
        h, w = webcam.height, webcam.width

        emit(status, "Loading model…")
        engine = RVMEngine(cfg.model_path, device, dtype, cfg.downsample)
        engine.warmup(h, w)

        preprocess = FramePreprocessor(device, dtype)
        compositor = Compositor(h, w)

        emit(status, "Building background…")
        static_bg, gpu_blur = build_background(cfg, h, w, device, dtype)

        return PipelineRuntime(
            device=device,
            dtype=dtype,
            gpu_name=gpu_name,
            webcam=webcam,
            engine=engine,
            preprocess=preprocess,
            compositor=compositor,
            static_bg=static_bg,
            gpu_blur=gpu_blur,
            height=h,
            width=w,
        )
    except Exception:
        if webcam is not None:
            try:
                webcam.release()
            except Exception:
                pass
        raise


def launch_gui(initial_config: PipelineConfig) -> None:
    try:
        from PySide6.QtCore import Qt, QThread, Signal, QTimer
        from PySide6.QtGui import QColor, QImage, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QButtonGroup,
            QColorDialog,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QFrame,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QRadioButton,
            QScrollArea,
            QSizePolicy,
            QSlider,
            QSpinBox,
            QStackedWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print(
            "ERROR: PySide6 is required for GUI mode.\n"
            "  Install it with:  pip install PySide6\n"
        )
        sys.exit(1)

    DARK_STYLE = """
    QMainWindow, QWidget {
        background-color: #1e1e2e;
        color: #cdd6f4;
        font-family: 'Segoe UI', 'Noto Sans', 'Ubuntu', sans-serif;
        font-size: 13px;
    }
    QGroupBox {
        border: 1px solid #45475a;
        border-radius: 6px;
        margin-top: 14px;
        padding: 12px 8px 8px 8px;
        font-weight: bold;
        color: #cba6f7;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
    }
    QLabel {
        color: #cdd6f4;
    }
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
        background-color: #313244;
        border: 1px solid #45475a;
        border-radius: 4px;
        padding: 4px 8px;
        color: #cdd6f4;
        min-height: 24px;
    }
    QComboBox:hover, QLineEdit:hover, QSpinBox:hover,
    QDoubleSpinBox:hover {
        border-color: #cba6f7;
    }
    QComboBox::drop-down {
        border: none;
        width: 24px;
    }
    QComboBox QAbstractItemView {
        background-color: #313244;
        color: #cdd6f4;
        selection-background-color: #585b70;
    }
    QPushButton {
        background-color: #45475a;
        border: 1px solid #585b70;
        border-radius: 5px;
        padding: 6px 18px;
        color: #cdd6f4;
        font-weight: bold;
        min-height: 28px;
    }
    QPushButton:hover {
        background-color: #585b70;
        border-color: #cba6f7;
    }
    QPushButton:pressed {
        background-color: #6c7086;
    }
    QPushButton:disabled {
        background-color: #313244;
        color: #6c7086;
        border-color: #45475a;
    }
    QPushButton#startBtn {
        background-color: #a6e3a1;
        color: #1e1e2e;
        border-color: #a6e3a1;
    }
    QPushButton#startBtn:hover {
        background-color: #b4f0b2;
    }
    QPushButton#startBtn:disabled {
        background-color: #3b5e3a;
        color: #6c7086;
    }
    QPushButton#stopBtn {
        background-color: #f38ba8;
        color: #1e1e2e;
        border-color: #f38ba8;
    }
    QPushButton#stopBtn:hover {
        background-color: #f5a0ba;
    }
    QPushButton#stopBtn:disabled {
        background-color: #5e3b45;
        color: #6c7086;
    }
    QRadioButton {
        color: #cdd6f4;
        spacing: 6px;
    }
    QRadioButton::indicator {
        width: 16px;
        height: 16px;
    }
    QSlider::groove:horizontal {
        height: 6px;
        background: #45475a;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #cba6f7;
        width: 16px;
        height: 16px;
        margin: -5px 0;
        border-radius: 8px;
    }
    QSlider::handle:horizontal:hover {
        background: #d9bfff;
    }
    QSlider::sub-page:horizontal {
        background: #cba6f7;
        border-radius: 3px;
    }
    QScrollArea {
        border: none;
    }
    QFrame#statusFrame {
        background-color: #181825;
        border: 1px solid #45475a;
        border-radius: 6px;
        padding: 4px;
    }
    QLabel#previewPlaceholder {
        color: #6c7086;
        font-size: 18px;
    }
    QLabel#fpsLabel {
        color: #a6e3a1;
        font-weight: bold;
        font-size: 14px;
    }
    QLabel#gpuLabel {
        color: #89b4fa;
    }
    QLabel#statusLabel {
        color: #f9e2af;
        font-weight: bold;
    }
    QPushButton#colorBtn {
        min-width: 60px;
        min-height: 28px;
        border-radius: 4px;
        border: 2px solid #585b70;
    }
    """

    class PipelineWorker(QThread):
        frame_ready = Signal(QImage)
        fps_updated = Signal(float)
        status_changed = Signal(str)
        error_occurred = Signal(str)
        log_message = Signal(str)

        def __init__(self, config: PipelineConfig, parent=None):
            super().__init__(parent)
            self.config = config
            self._running = False
            self._lock = threading.Lock()
            self._pending_bg: Optional[Dict[str, Any]] = None
            self._pending_downsample: Optional[float] = None

        def request_bg_change(self, **changes: Any) -> None:
            with self._lock:
                self._pending_bg = changes

        def request_downsample_change(self, ratio: float) -> None:
            with self._lock:
                self._pending_downsample = ratio

        def stop(self) -> None:
            self._running = False

        def run(self) -> None:
            cfg = self.config
            runtime = None
            vcam = None
            self._running = True

            try:
                runtime = init_runtime(
                    cfg,
                    status=self.status_changed.emit,
                    log=self.log_message.emit,
                )

                self.status_changed.emit("Opening virtual camera…")
                vcam = pyvirtualcam.Camera(
                    width=runtime.width,
                    height=runtime.height,
                    fps=cfg.fps,
                    fmt=PixelFormat.RGB,
                    device=cfg.vcam,
                )
                self.log_message.emit(f"[vCam] Active: {vcam.device}")

                fps_counter = FPSCounter(log_interval=2.0)
                last_preview = 0.0
                self.status_changed.emit("Running")

                while self._running:
                    with self._lock:
                        bg_changes = self._pending_bg
                        downsample = self._pending_downsample
                        self._pending_bg = None
                        self._pending_downsample = None

                    if bg_changes is not None:
                        try:
                            runtime.static_bg, runtime.gpu_blur = (
                                rebuild_background(
                                    cfg,
                                    runtime.height,
                                    runtime.width,
                                    runtime.device,
                                    runtime.dtype,
                                    **bg_changes,
                                )
                            )
                        except Exception as exc:
                            self.error_occurred.emit(
                                f"Background change failed: {exc}"
                            )

                    if downsample is not None:
                        cfg.downsample = downsample
                        runtime.engine.downsample_ratio = downsample
                        runtime.engine.reset()

                    frame_bgr = runtime.webcam.wait_and_read()
                    if frame_bgr is None:
                        continue

                    src = runtime.preprocess(frame_bgr)
                    bg = current_background(
                        src, runtime.static_bg, runtime.gpu_blur
                    )
                    fgr, pha = runtime.engine(src)
                    frame_rgb = runtime.compositor(fgr, pha, bg)

                    vcam.send(frame_rgb)
                    vcam.sleep_until_next_frame()

                    now = time.monotonic()
                    if now - last_preview >= PREVIEW_INTERVAL_S:
                        last_preview = now
                        h, w, ch = frame_rgb.shape
                        qimg = QImage(
                            frame_rgb.data,
                            w,
                            h,
                            ch * w,
                            QImage.Format.Format_RGB888,
                        ).copy()
                        self.frame_ready.emit(qimg)

                    fps = fps_counter.tick()
                    if fps is not None:
                        self.fps_updated.emit(fps)

            except Exception as exc:
                self.error_occurred.emit(str(exc))
            finally:
                self.status_changed.emit("Stopping…")
                if vcam is not None:
                    try:
                        vcam.close()
                    except Exception:
                        pass
                close_runtime(runtime)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.status_changed.emit("Stopped")

    class PreviewLabel(QLabel):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._pixmap = None
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setMinimumSize(320, 180)
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self.setStyleSheet(
                "background-color: #11111b; border-radius: 8px;"
            )

        def set_image(self, qimg: QImage) -> None:
            self._pixmap = QPixmap.fromImage(qimg)
            self._rescale()

        def clear_image(self) -> None:
            self._pixmap = None
            self.setText("No Signal")
            self.setObjectName("previewPlaceholder")

        def resizeEvent(self, event) -> None:
            super().resizeEvent(event)
            self._rescale()

        def _rescale(self) -> None:
            if self._pixmap and not self._pixmap.isNull():
                super().setPixmap(
                    self._pixmap.scaled(
                        self.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.FastTransformation,
                    )
                )

    class MainWindow(QMainWindow):
        def __init__(self, initial_config: PipelineConfig):
            super().__init__()
            self.setWindowTitle("RVM Background Studio")
            self.setMinimumSize(820, 760)
            self.resize(960, 880)

            self.cfg = initial_config
            self.worker: Optional[PipelineWorker] = None

            self._res_fps_map: Dict[Tuple[int, int], List[int]] = {}

            self._debounce_timer = QTimer(self)
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.setInterval(200)
            self._debounce_timer.timeout.connect(self._apply_bg_change)

            self._build_ui()
            self._connect_signals()
            self._populate_devices()
            self._load_config_into_ui()

        def _build_ui(self) -> None:
            central = QWidget()
            self.setCentralWidget(central)

            root = QVBoxLayout(central)
            root.setSpacing(8)
            root.setContentsMargins(10, 10, 10, 10)

            self.preview = PreviewLabel()
            self.preview.clear_image()
            root.addWidget(self.preview, stretch=3)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)

            ctrl_widget = QWidget()
            ctrl_layout = QVBoxLayout(ctrl_widget)
            ctrl_layout.setSpacing(8)
            scroll.setWidget(ctrl_widget)
            root.addWidget(scroll, stretch=2)

            dev_grp = QGroupBox("Device Settings")
            dev_form = QFormLayout(dev_grp)

            self.combo_webcam = QComboBox()
            dev_form.addRow("Webcam:", self.combo_webcam)

            self.edit_vcam = QLineEdit()
            dev_form.addRow("Virtual Cam:", self.edit_vcam)

            self.combo_resolution = QComboBox()
            dev_form.addRow("Resolution:", self.combo_resolution)

            self.combo_fps = QComboBox()
            dev_form.addRow("FPS:", self.combo_fps)

            ctrl_layout.addWidget(dev_grp)

            model_grp = QGroupBox("Model Settings")
            model_form = QFormLayout(model_grp)

            model_row = QHBoxLayout()
            self.edit_model = QLineEdit()
            model_row.addWidget(self.edit_model, stretch=1)
            self.btn_browse_model = QPushButton("Browse…")
            model_row.addWidget(self.btn_browse_model)
            model_form.addRow("Model:", model_row)

            ds_row = QHBoxLayout()
            self.slider_downsample = QSlider(Qt.Orientation.Horizontal)
            self.slider_downsample.setRange(5, 100)
            self.slider_downsample.setTickInterval(5)
            ds_row.addWidget(self.slider_downsample, stretch=1)

            self.lbl_downsample = QLabel("0.25")
            self.lbl_downsample.setMinimumWidth(40)
            ds_row.addWidget(self.lbl_downsample)
            model_form.addRow("Downsample:", ds_row)

            ctrl_layout.addWidget(model_grp)

            bg_grp = QGroupBox("Background")
            bg_layout = QVBoxLayout(bg_grp)

            radio_row = QHBoxLayout()
            self.radio_blur = QRadioButton("Blur")
            self.radio_green = QRadioButton("Green Screen")
            self.radio_image = QRadioButton("Image")
            self.radio_color = QRadioButton("Solid Color")
            self.bg_radios = (
                self.radio_blur,
                self.radio_green,
                self.radio_image,
                self.radio_color,
            )
            self.bg_radio_group = QButtonGroup(self)
            for i, rb in enumerate(self.bg_radios):
                self.bg_radio_group.addButton(rb, i)
                radio_row.addWidget(rb)
            bg_layout.addLayout(radio_row)

            self.bg_stack = QStackedWidget()

            # blur
            blur_page = QWidget()
            blur_form = QFormLayout(blur_page)
            blur_form.setContentsMargins(0, 4, 0, 0)

            kr_row = QHBoxLayout()
            self.slider_blur_radius = QSlider(Qt.Orientation.Horizontal)
            self.slider_blur_radius.setRange(1, 151)
            kr_row.addWidget(self.slider_blur_radius, stretch=1)
            self.lbl_blur_radius = QLabel("51")
            self.lbl_blur_radius.setMinimumWidth(36)
            kr_row.addWidget(self.lbl_blur_radius)
            blur_form.addRow("Kernel:", kr_row)

            self.spin_blur_sigma = QDoubleSpinBox()
            self.spin_blur_sigma.setRange(0.0, 100.0)
            self.spin_blur_sigma.setSingleStep(0.5)
            self.spin_blur_sigma.setSpecialValueText("auto")
            blur_form.addRow("Sigma:", self.spin_blur_sigma)

            bds_row = QHBoxLayout()
            self.slider_blur_ds = QSlider(Qt.Orientation.Horizontal)
            self.slider_blur_ds.setRange(1, 8)
            bds_row.addWidget(self.slider_blur_ds, stretch=1)
            self.lbl_blur_ds = QLabel("4")
            self.lbl_blur_ds.setMinimumWidth(20)
            bds_row.addWidget(self.lbl_blur_ds)
            blur_form.addRow("Downscale:", bds_row)

            self.bg_stack.addWidget(blur_page)

            # green
            green_page = QWidget()
            green_layout = QVBoxLayout(green_page)
            green_layout.addWidget(
                QLabel("Standard chroma-key green (#00B100).")
            )
            self.bg_stack.addWidget(green_page)

            # image
            image_page = QWidget()
            image_form = QFormLayout(image_page)
            image_form.setContentsMargins(0, 4, 0, 0)

            img_row = QHBoxLayout()
            self.edit_bg_image = QLineEdit()
            self.edit_bg_image.setPlaceholderText("Select an image…")
            img_row.addWidget(self.edit_bg_image, stretch=1)
            self.btn_browse_bg = QPushButton("Browse…")
            img_row.addWidget(self.btn_browse_bg)
            image_form.addRow("Image:", img_row)

            self.bg_stack.addWidget(image_page)

            # solid color
            color_page = QWidget()
            color_form = QFormLayout(color_page)
            color_form.setContentsMargins(0, 4, 0, 0)

            self.btn_color = QPushButton()
            self.btn_color.setObjectName("colorBtn")
            self._current_color = QColor(*DEFAULT_BG_COLOR)
            self._update_color_button()
            color_form.addRow("Color:", self.btn_color)

            self.bg_stack.addWidget(color_page)

            bg_layout.addWidget(self.bg_stack)
            ctrl_layout.addWidget(bg_grp)

            # Status bar
            status_frame = QFrame()
            status_frame.setObjectName("statusFrame")
            status_layout = QHBoxLayout(status_frame)
            status_layout.setContentsMargins(10, 4, 10, 4)

            self.lbl_fps = QLabel("FPS: —")
            self.lbl_fps.setObjectName("fpsLabel")
            status_layout.addWidget(self.lbl_fps)

            status_layout.addStretch()

            self.lbl_gpu = QLabel("GPU: detecting…")
            self.lbl_gpu.setObjectName("gpuLabel")
            status_layout.addWidget(self.lbl_gpu)

            status_layout.addStretch()

            self.lbl_status = QLabel("Stopped")
            self.lbl_status.setObjectName("statusLabel")
            status_layout.addWidget(self.lbl_status)

            root.addWidget(status_frame)

            # Start / Stop
            btn_row = QHBoxLayout()
            btn_row.addStretch()

            self.btn_start = QPushButton("▶  Start")
            self.btn_start.setObjectName("startBtn")
            self.btn_start.setMinimumWidth(120)
            btn_row.addWidget(self.btn_start)

            self.btn_stop = QPushButton("■  Stop")
            self.btn_stop.setObjectName("stopBtn")
            self.btn_stop.setMinimumWidth(120)
            self.btn_stop.setEnabled(False)
            btn_row.addWidget(self.btn_stop)

            btn_row.addStretch()
            root.addLayout(btn_row)

            try:
                _, _, gpu_name = setup_gpu()
                self.lbl_gpu.setText(f"GPU: {gpu_name}")
            except Exception:
                self.lbl_gpu.setText("GPU: N/A")

        # signals
        def _connect_signals(self) -> None:
            self.btn_start.clicked.connect(self._on_start)
            self.btn_stop.clicked.connect(self._on_stop)
            self.btn_browse_model.clicked.connect(self._browse_model)
            self.btn_browse_bg.clicked.connect(self._browse_bg_image)
            self.btn_color.clicked.connect(self._pick_color)

            self.combo_webcam.currentIndexChanged.connect(
                self._on_webcam_changed
            )
            self.combo_resolution.currentIndexChanged.connect(
                self._on_resolution_changed
            )

            self.bg_radio_group.idToggled.connect(self._on_bg_radio)
            self.slider_blur_radius.valueChanged.connect(
                self._on_blur_slider
            )
            self.slider_blur_ds.valueChanged.connect(
                self._on_blur_ds_slider
            )
            self.spin_blur_sigma.valueChanged.connect(
                self._schedule_bg_change
            )
            self.slider_downsample.valueChanged.connect(
                self._on_downsample_slider
            )
            self.edit_bg_image.editingFinished.connect(
                self._schedule_bg_change
            )

        # device population
        def _populate_devices(self) -> None:
            self.combo_webcam.blockSignals(True)
            self.combo_webcam.clear()
            for dev in enumerate_v4l2_devices():
                self.combo_webcam.addItem(
                    f"/dev/video{dev['index']}  —  {dev['name']}",
                    dev["index"],
                )

            for i in range(self.combo_webcam.count()):
                if self.combo_webcam.itemData(i) == self.cfg.webcam:
                    self.combo_webcam.setCurrentIndex(i)
                    break
            self.combo_webcam.blockSignals(False)

            # Trigger initial resolution query for the selected camera
            self._query_and_populate_resolutions()

        def _on_webcam_changed(self, _index: int) -> None:
            """Camera selection changed – re-query its capabilities."""
            self._query_and_populate_resolutions()

        def _query_and_populate_resolutions(self) -> None:
            """Ask the currently selected camera what it supports and fill
            the Resolution and FPS combo-boxes accordingly."""
            cam_idx = self.combo_webcam.currentData()
            if cam_idx is None:
                cam_idx = 0

            resolutions = get_preferred_resolutions(cam_idx)

            # Build lookup map  (w, h) → [fps …]
            self._res_fps_map = {
                (w, h): fps_list for w, h, fps_list in resolutions
            }

            self.combo_resolution.blockSignals(True)
            self.combo_resolution.clear()
            for w, h, _fps in resolutions:
                self.combo_resolution.addItem(f"{w} × {h}", (w, h))

            # Try to pre-select the resolution closest to the current config
            best_idx = 0
            best_dist = float("inf")
            target_pixels = self.cfg.width * self.cfg.height
            for i in range(self.combo_resolution.count()):
                rw, rh = self.combo_resolution.itemData(i)
                dist = abs(rw * rh - target_pixels)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            self.combo_resolution.setCurrentIndex(best_idx)
            self.combo_resolution.blockSignals(False)

            self._populate_fps_for_current_resolution()

        def _on_resolution_changed(self, _index: int) -> None:
            self._populate_fps_for_current_resolution()

        def _populate_fps_for_current_resolution(self) -> None:
            res = self.combo_resolution.currentData()
            if res is None:
                return
            fps_list = self._res_fps_map.get(res, [30])

            self.combo_fps.blockSignals(True)
            self.combo_fps.clear()
            for f in fps_list:
                self.combo_fps.addItem(f"{f} fps", f)

            best_idx = 0
            best_dist = float("inf")
            for i in range(self.combo_fps.count()):
                dist = abs(self.combo_fps.itemData(i) - self.cfg.fps)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            self.combo_fps.setCurrentIndex(best_idx)
            self.combo_fps.blockSignals(False)

        def _load_config_into_ui(self) -> None:
            self.edit_vcam.setText(self.cfg.vcam)
            self.edit_model.setText(self.cfg.model_path)

            self.slider_downsample.setValue(int(self.cfg.downsample * 100))
            self.lbl_downsample.setText(f"{self.cfg.downsample:.2f}")

            self.slider_blur_radius.setValue((self.cfg.blur_radius - 1) // 2)
            self.lbl_blur_radius.setText(str(self.cfg.blur_radius))
            self.spin_blur_sigma.setValue(self.cfg.blur_sigma)
            self.slider_blur_ds.setValue(self.cfg.blur_downscale)
            self.lbl_blur_ds.setText(str(self.cfg.blur_downscale))

            idx = bg_mode_to_id(self.cfg.bg_mode)
            self.bg_radios[idx].setChecked(True)
            self.bg_stack.setCurrentIndex(idx)

            if self.cfg.bg_image:
                self.edit_bg_image.setText(self.cfg.bg_image)

            self._current_color = QColor(*self.cfg.bg_color)
            self._update_color_button()

        def _read_config(self) -> PipelineConfig:
            webcam_idx = self.combo_webcam.currentData()
            if webcam_idx is None:
                webcam_idx = 0

            res = self.combo_resolution.currentData()
            if res is not None:
                w, h = res
            else:
                w, h = DEFAULT_WIDTH, DEFAULT_HEIGHT

            fps = self.combo_fps.currentData()
            if fps is None:
                fps = DEFAULT_FPS

            color = self._current_color
            return PipelineConfig(
                model_path=self.edit_model.text(),
                webcam=webcam_idx,
                vcam=self.edit_vcam.text(),
                width=w,
                height=h,
                fps=fps,
                downsample=self.slider_downsample.value() / 100.0,
                bg_mode=bg_mode_from_id(self.bg_radio_group.checkedId()),
                bg_image=self.edit_bg_image.text() or None,
                bg_color=(color.red(), color.green(), color.blue()),
                blur_radius=self.slider_blur_radius.value() * 2 + 1,
                blur_sigma=self.spin_blur_sigma.value(),
                blur_downscale=self.slider_blur_ds.value(),
            )

        def _current_bg_changes(self) -> Dict[str, Any]:
            color = self._current_color
            return dict(
                bg_mode=bg_mode_from_id(self.bg_radio_group.checkedId()),
                bg_image=self.edit_bg_image.text() or None,
                bg_color=(color.red(), color.green(), color.blue()),
                blur_radius=self.slider_blur_radius.value() * 2 + 1,
                blur_sigma=self.spin_blur_sigma.value(),
                blur_downscale=self.slider_blur_ds.value(),
            )

        def _on_start(self) -> None:
            cfg = self._read_config()

            if not Path(cfg.model_path).is_file():
                QMessageBox.critical(
                    self,
                    "Model Not Found",
                    f"Model file not found:\n{cfg.model_path}\n\n"
                    "Download from:\nhttps://github.com/PeterL1n/"
                    "RobustVideoMatting/releases",
                )
                return

            if cfg.bg_mode == "image" and (
                not cfg.bg_image or not Path(cfg.bg_image).is_file()
            ):
                QMessageBox.warning(
                    self,
                    "No Background Image",
                    "Please select a valid background image.",
                )
                return

            self._set_running_state(True)
            self.preview.clear_image()
            self.preview.setText("Starting…")
            self.lbl_fps.setText("FPS: —")

            self.worker = PipelineWorker(cfg)
            self.worker.frame_ready.connect(self._update_preview)
            self.worker.fps_updated.connect(self._update_fps)
            self.worker.status_changed.connect(self._update_status)
            self.worker.error_occurred.connect(self._on_error)
            self.worker.log_message.connect(self._on_log)
            self.worker.finished.connect(self._on_worker_finished)
            self.worker.start()

        def _on_stop(self) -> None:
            if self.worker:
                self.worker.stop()
                self.btn_stop.setEnabled(False)
                self.lbl_status.setText("Stopping…")

        def _on_worker_finished(self) -> None:
            if self.worker:
                self.worker.deleteLater()
                self.worker = None
            self._set_running_state(False)
            self.preview.clear_image()

        def _set_running_state(self, running: bool) -> None:
            self.btn_start.setEnabled(not running)
            self.btn_stop.setEnabled(running)
            for widget in (
                self.combo_webcam,
                self.edit_vcam,
                self.combo_resolution,
                self.combo_fps,
                self.edit_model,
                self.btn_browse_model,
            ):
                widget.setEnabled(not running)

        def _update_preview(self, qimg: QImage) -> None:
            self.preview.set_image(qimg)

        def _update_fps(self, fps: float) -> None:
            self.lbl_fps.setText(f"FPS: {fps:.1f}")

        def _update_status(self, text: str) -> None:
            self.lbl_status.setText(text)

        def _on_error(self, msg: str) -> None:
            QMessageBox.critical(self, "Pipeline Error", msg)

        def _on_log(self, msg: str) -> None:
            print(msg)

        def _browse_model(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select RVM Model",
                str(Path.home()),
                "TorchScript (*.torchscript);;All Files (*)",
            )
            if path:
                self.edit_model.setText(path)

        def _browse_bg_image(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Background Image",
                str(Path.home()),
                "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)",
            )
            if path:
                self.edit_bg_image.setText(path)
                self._schedule_bg_change()

        def _pick_color(self) -> None:
            color = QColorDialog.getColor(
                self._current_color,
                self,
                "Background Color",
            )
            if color.isValid():
                self._current_color = color
                self._update_color_button()
                self._schedule_bg_change()

        def _update_color_button(self) -> None:
            c = self._current_color
            lum = 0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()
            fg = "#000" if lum > 128 else "#fff"
            self.btn_color.setStyleSheet(
                f"QPushButton#colorBtn {{ background-color: {c.name()}; "
                f"color: {fg}; border: 2px solid #585b70; "
                f"border-radius: 4px; min-width: 60px; "
                f"min-height: 28px; }}"
            )
            self.btn_color.setText(c.name())

        def _on_bg_radio(self, idx: int, checked: bool) -> None:
            if checked:
                self.bg_stack.setCurrentIndex(idx)
                self._schedule_bg_change()

        def _on_blur_slider(self, val: int) -> None:
            self.lbl_blur_radius.setText(str(val * 2 + 1))
            self._schedule_bg_change()

        def _on_blur_ds_slider(self, val: int) -> None:
            self.lbl_blur_ds.setText(str(val))
            self._schedule_bg_change()

        def _on_downsample_slider(self, val: int) -> None:
            ratio = val / 100.0
            self.lbl_downsample.setText(f"{ratio:.2f}")
            if self.worker and self.worker.isRunning():
                self.worker.request_downsample_change(ratio)

        def _schedule_bg_change(self, *_args) -> None:
            self._debounce_timer.start()

        def _apply_bg_change(self) -> None:
            if self.worker and self.worker.isRunning():
                self.worker.request_bg_change(**self._current_bg_changes())

        def closeEvent(self, event) -> None:
            if self.worker and self.worker.isRunning():
                self.worker.stop()
                self.worker.wait(5000)
            event.accept()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    win = MainWindow(initial_config)
    win.show()
    sys.exit(app.exec())


def run_cli(cfg: PipelineConfig, preview: bool = False) -> None:
    if not Path(cfg.model_path).is_file():
        print(f"ERROR: Model not found at {cfg.model_path}")
        print("Run setup.sh or download from:")
        print("https://github.com/PeterL1n/RobustVideoMatting/releases")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    runtime = None
    vcam = None

    try:
        runtime = init_runtime(cfg)

        major, minor = torch.cuda.get_device_capability(runtime.device)
        if major >= 7:
            print(f"[GPU] {runtime.gpu_name} (sm_{major}{minor}) -> FP16")
        else:
            print(
                f"[GPU] {runtime.gpu_name} (sm_{major}{minor}) -> FP32 "
                "(pre-Turing)"
            )
            print(
                "      Consider rvm_resnet50_fp32.torchscript for this GPU."
            )

        if runtime.gpu_blur is not None:
            sigma = resolve_blur_sigma(cfg.blur_radius, cfg.blur_sigma)
            print(
                f"[Blur] GPU Gaussian blur: kernel={cfg.blur_radius}, "
                f"sigma={sigma:.2f}, downscale={cfg.blur_downscale}x"
            )

        print(
            f"[vCam] Opening {cfg.vcam} "
            f"({runtime.width}x{runtime.height} @ {cfg.fps}fps)"
        )
        vcam = pyvirtualcam.Camera(
            width=runtime.width,
            height=runtime.height,
            fps=cfg.fps,
            fmt=PixelFormat.RGB,
            device=cfg.vcam,
        )
        print(f"[vCam] Active: {vcam.device}")

        fps_counter = FPSCounter(log_interval=5.0)

        print(f"\n{'=' * 60}")
        print("  LIVE — Ctrl+C to stop")
        print(
            f"  Mode: {cfg.bg_mode} | "
            f"{runtime.width}x{runtime.height} @ {cfg.fps}fps"
        )
        if preview:
            print("  Preview: enabled (press 'q' in window to quit)")
        print(f"{'=' * 60}\n")

        while True:
            frame_bgr = runtime.webcam.wait_and_read()
            if frame_bgr is None:
                continue

            src = runtime.preprocess(frame_bgr)
            bg = current_background(src, runtime.static_bg, runtime.gpu_blur)
            fgr, pha = runtime.engine(src)
            frame_rgb = runtime.compositor(fgr, pha, bg)

            vcam.send(frame_rgb)
            vcam.sleep_until_next_frame()

            if preview:
                cv2.imshow(
                    "RVM Preview",
                    cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            fps = fps_counter.tick()
            if fps is not None:
                print(f"[FPS] {fps:.1f}")

    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user.")
    finally:
        if vcam is not None:
            try:
                vcam.close()
            except Exception:
                pass
        close_runtime(runtime)
        if preview:
            cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RVM ResNet50 real-time webcam matting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:

  # Launch GUI:
  python run.py --gui

  # List camera capabilities then exit:
  python run.py --list-formats

  # Blur background (CLI, default):
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

    parser.add_argument(
        "--gui", action="store_true", help="Launch the graphical interface"
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="Print every format / resolution / FPS the webcam supports, "
        "then exit",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--webcam", type=int, default=DEFAULT_WEBCAM)
    parser.add_argument("--vcam", default=DEFAULT_VCAM)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--downsample", type=float, default=DEFAULT_DOWNSAMPLE)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a local OpenCV preview window (CLI mode)",
    )

    bg = parser.add_argument_group("Background")
    bg.add_argument("--bg-mode", default="blur", choices=BG_MODES)
    bg.add_argument("--bg-image", default=None)
    bg.add_argument(
        "--bg-color", default=DEFAULT_BG_COLOR_STR, help="R,G,B 0-255"
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
        help="Downscale factor for fast blur (1 = full-res, 4 = 4x faster)",
    )

    args = parser.parse_args()

    if args.list_formats:
        print(f"Querying /dev/video{args.webcam} …\n")
        print_camera_formats(args.webcam)
        sys.exit(0)

    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
        return

    if args.gui:
        launch_gui(cfg)
    else:
        run_cli(cfg, preview=args.preview)


if __name__ == "__main__":
    main()
