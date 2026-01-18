import os
from typing import cast
import numpy as np
import onnxruntime as rt
import argparse
from model import RVMInference
from turbojpeg import TJPF_RGB, TurboJPEG
from utils import black_filter, color_filter, blur_filter, get_output_cam, hwc_to_bchw, start_camera

jpeg = TurboJPEG()

def handle_resolution_arg(arg: str) -> tuple[int, int]:
    split = arg.split(",")
    if split[0] == arg:
        print("Error: You are missing a comma in the resolution argument")
        exit()
    if len(split) != 2:
        print("Error: Too many commas in resolution argument")
        exit()
    try:
        f, l = int(split[0]), int(split[1])
    except:
        print("Error: Could not convert resolution argument to integers...")
        exit()
    finally:
        return (f,l)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog='BackgroundRemoval4Linux',
            description="Removes the background of a camera stream and outputs to a virtual camera using v4l2"
            )

    parser.add_argument('path', help="Absolute (starting with /) or relative path of the .pth ONNX RVM file")
    parser.add_argument('video_output', help="The integer number of your video output device as found when running: v4l2-ctl --list-devices.Note: this must be under /dev/video[number], not media or anything else.", type=int)
    parser.add_argument('--downsample_ratio', help="Downsample ratio lower values help with performance but degrade quality, default = 1.0, i.e. no downsampling", type=float, default=1.0)
    parser.add_argument('--filter', help="Type of filter to be applied", default='blur', nargs='?', choices=['black', 'green', 'blue', 'blur'])
    parser.add_argument('--blur_strength', help="Can only be used if blur is the selected filter", type=int)
    parser.add_argument('--resolution', help="Exact resolution of the camera in width,height format (your camera must be able to support this) run v4l2-ctl --list-framesizes=MJPG to verify support", type=str)
    parser.add_argument('--fps', help="Exact FPS of the camera (you can write 30 instead of 30.000) run v4l2-ctl --list-formats-ext to verify support", type=float)

    args = parser.parse_args()
    if not os.path.isfile(args.path):
        print("Error: path provided does not exist...")
        exit()

    if args.blur_strength and args.filter != "blur":
        print("Error: You cannot set blur strength if the selected filter is not blur")
        exit()

    if isinstance(args.blur_strength, int):
        if args.blur_strength % 2 == 0:
            args.blur_strength += 1

    cam_res: tuple[None|int, None|int] = (None,None)
    if isinstance(args.resolution, str):
        cam_res = handle_resolution_arg(args.resolution)

    fps: None|float = None
    if isinstance(args.fps, float):
        fps = args.fps

    (cam, cap) = start_camera(fps, cam_res[0], cam_res[1])
    format = cap.get_format()
    (dev_out, out) = get_output_cam(args.video_output, format.width, format.height, float(cap.get_fps()))

    bg_arr: None|np.ndarray = None
    if args.filter in ['black', 'green', 'blue']:
        bg_arr = np.zeros((format.height, format.width, 3), dtype=np.uint8)

    options = rt.SessionOptions()
    options.log_severity_level = 1 
    providers = [
    'CUDAExecutionProvider',
    'MIGraphXExecutionProvider',
    'CPUExecutionProvider']

    session = rt.InferenceSession(args.path, providers=providers, sess_options=options)
    print(f"Running on: {session.get_providers()[0]}")
    rvm = RVMInference(session)

    with cap, out:
        for frame in cap:
            rgb_u8 = jpeg.decode(frame.array, pixel_format=TJPF_RGB)
            rgb_f32 = rgb_u8.astype(np.float32) / 255.0
            bchw = hwc_to_bchw(rgb_f32)

            # in hwc
            fgr, alpha = rvm.process(bchw, downsample_ratio=args.downsample_ratio)
            res = np.zeros(1)
            match args.filter:
                case 'green':
                    bg_arr = cast(np.ndarray, bg_arr)
                    bg_arr.fill(0)
                    bg_arr[:,:,1] = 255
                    res = color_filter(fgr, alpha, bg_arr)
                case 'blue':
                    bg_arr = cast(np.ndarray, bg_arr)
                    bg_arr.fill(0)
                    bg_arr[:,:,2] = 255
                    res = color_filter(fgr, alpha, bg_arr)
                case 'black':
                    res = black_filter(fgr, alpha)
                case 'blur':
                    if isinstance(args.blur_strength, int):
                        res = blur_filter(fgr, alpha, rgb_u8, args.blur_strength)
                    else:
                        res = blur_filter(fgr, alpha, rgb_u8)

            res = np.ascontiguousarray(res).astype(np.uint8)
            jpg_bytes = jpeg.encode(res, quality=85, pixel_format=TJPF_RGB)
            out.write(jpg_bytes)

    cam.close()
    dev_out.close()
