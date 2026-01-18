# BackgroundRemoval4Linux

*Made because I've given up on waiting for NVIDIA Broadcast to come to Linux.*

**Note: Due to the current code doing copies from the CPU & RAM and not utilising the VRAM it does not perform at near the expeceted for GPU, rather you will be bottle-necked by the CPU. This is still under moderate development so I plan to fix this.**

## Requirements
Install the following:
* Python 3.13
* v4l2loopback

### GPU

For NVIDIA GPU inference you need:
- CUDA 12.*
- cuDNN 9.* 

You may also choose to use TensorRT instead.

For AMD GPU inference you need:
- MIGraphX ROCm 7.0

More detailed information can be found on this [page](https://onnxruntime.ai/docs/execution-providers/)

### v4l2loopback

Make sure you have a v4l2loopback device, you can make one with the following command:

```bash
sudo modprobe v4l2loopback video_nr=[int] card_label="RVM Output"
```

To avoid clashes you should check that the number you choose isn't already taken by running:
```bash
v4l2-ctl --list-devices
```
You can run this after making the device again to see your device appear under `/dev/video[number you chose]`

The above command only works temporarily and must be repeated per reboot.

After you've settled on a configuration you may want to make this change permanent. You can do so as follows:
```bash
sudo [fav_text_editor] /etc/modprobe.d/v4l2loopback.conf
```

add the following line:
```
options v4l2loopback video_nr=[int] card_label=[string label] 
```

Download an RVM model from the [Github](https://github.com/PeterL1n/RobustVideoMatting#download). Only download models from the ONNX section, ResNet is higher quality than MobileNet.

## Usage
Setup a python virtual environment and install required packages:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

At its most basic the script is run using a model path and device number of the v4l2loopback device:
```bash
python main.py [model_path] [device_number]
```

There are other options like using a specific resolution & fps from your camera via:
```bash
python main.py [model_path] [device_number] --resolution=640,480 --fps=30
```

Or adding a downsample ratio, which is useful for performance. It will degrade the quality of the background mask but not of the resolution of the video.
```bash
python main.py [model_path] [device_number] --downsample_ratio=0.4
```

Use `--help` to view all the possible options.

## Mamba
If you have a later version of CUDA/cudNN installed that is not supported by the ONNXRuntime you will not be able to run the GPU inference.

To fix this you have a few solutions:
- Downgrade versions to match those specified in the error message you will recieve.
- Install NVIDIA's CUDA & cuDNN binaries from PyPI, i.e., `pip install nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12`
- Use Mamba which the following guide will describe:

-- under construction -- 

*Licensed under GPL-3.0 due to the usage of [linuxpy](https://github.com/tiagocoutinho/linuxpy/)*
