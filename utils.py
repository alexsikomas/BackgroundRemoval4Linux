from linuxpy.video.device import Device, InfoEx, PixelFormat, VideoCapture, VideoOutput
from operator import itemgetter
from typing import Optional, cast
import numpy as np
import cv2 as cv

type FrameInfo = tuple[int, int, float]

# only supports Motion JPEG
# 0 means to match the highest fps and res you can find
def start_camera(target_fps: Optional[float], target_w: Optional[int], target_h: Optional[int]) -> tuple[Device, VideoCapture]:
    cam = Device.from_id(0)
    cam.open()

    if not isinstance(cam.info, InfoEx):
        print("Error: Issue getting camera information")
        exit()

    # format [(width, height, fps))]
    mjpeg_idxs: list[int] = []
    mjpeg_fts: list[FrameInfo] = []
    for i, ft in enumerate(cam.info.frame_sizes):
        px_f = cast(PixelFormat, ft.pixel_format)
        if px_f.name == "MJPEG":
            fps = float(ft.max_fps)
            width = cast(int, ft.width)
            height = cast(int, ft.height)
            mjpeg_idxs.append(i)
            mjpeg_fts.append((width, height, fps))

    idx = get_best_mode(target_fps, target_w, target_h, mjpeg_fts)
    if idx == -1:
        print("Error: cannot find a match for the camera fps & resolution in Motion-JPEG format...")
        exit()

    matched_frame = cam.info.frame_sizes[mjpeg_idxs[idx]]
    cap = VideoCapture(cam)
    print("FRAME SIZE: ", matched_frame.width, "x", matched_frame.height)
    cap.set_format(matched_frame.width, matched_frame.height, "MJPG")
    cap.set_fps(float(matched_frame.max_fps))
    return (cam, cap)

def get_output_cam(num: int, width: int, height: int, fps: float) -> tuple[Device, VideoOutput]:
    dev_out = Device.from_id(num)
    dev_out.open()
    out = VideoOutput(dev_out)
    out.set_format(width, height, "MJPG")
    out.set_fps(fps)
    return (dev_out, out)

# returns the index of the best mode
def get_best_mode(fps: Optional[float], w: Optional[int], h: Optional[int], fts: list[FrameInfo]) -> int:
    match (fps, w, h):
        case (float(), int(), int()):
            if (w,h,fps) in fts:
                return fts.index((w,h,fps))
        case (float(), None, None):
            matched_indicies = [i for i, ft in enumerate(fts) if ft[2] == fps]
            matched_fts = list(itemgetter(*matched_indicies)(fts))
            if isinstance(matched_fts[0], tuple):
                matched_frame = heuristic_match(matched_fts)
                return fts.index(matched_frame)
            else:
                return matched_indicies[0]
        case (None, int(), int()):
            matched_indicies = [i for i, ft in enumerate(fts) if (ft[0] == w and ft[1] == h)]
            matched_fts = list(itemgetter(*matched_indicies)(fts))
            if isinstance(matched_fts[0], tuple):
                matched_frame = heuristic_match(matched_fts)
                return fts.index(matched_frame)
            else:
                return matched_indicies[0]
        case (None, None, None):
            matched_frame = heuristic_match(fts)
            return fts.index(matched_frame)

    return -1

def heuristic_match(fts: list[FrameInfo]):
    fps_weight, res_weight = 0.7, 0.3
    max_fps = max(ft[2] for ft in fts)
    max_res = max(ft[0] * ft[1] for ft in fts)
    def score(ft):
        return (fps_weight * (ft[2]/max_fps)) + (res_weight * ((ft[0]*ft[1])/max_res))

    return max(fts, key=score)

# flips b & r channel if in rgb or bgr form
def flip_b_r_channel(input: np.ndarray) -> np.ndarray:
    # switch b & r
    if input.ndim != 3:
        print("Incorrect dimension size while converting to rgb!")
        exit()

    return input[..., ::-1]

# h = height, w = width, c = channel, b = batch
def hwc_to_bchw(input: np.ndarray) -> np.ndarray:
    arr = input.transpose(2, 0, 1)
    arr = arr[np.newaxis, ...]
    # contig in ram
    return np.ascontiguousarray(arr)

def bchw_to_hwc(input: np.ndarray) -> np.ndarray:
    arr = input[0]
    arr = np.transpose(arr, (1, 2, 0))
    return arr

def black_filter(fgr: np.ndarray, alpha: np.ndarray):
    return (fgr.astype(np.uint16) * alpha) // 255

def color_filter(fgr: np.ndarray, alpha: np.ndarray, bg: np.ndarray):
    inv_alpha = 255 - alpha
    return (fgr.astype(np.uint16)*alpha + (bg.astype(np.uint16)*inv_alpha)) // 255

def blur_filter(fgr: np.ndarray, alpha: np.ndarray, orig: np.ndarray, strength=51):
    blurred_bg = cv.GaussianBlur(orig, (strength, strength), 0)
    inv_alpha = 255 - alpha
    return (fgr.astype(np.uint16) * alpha + blurred_bg.astype(np.uint16) * inv_alpha) // 255
