*Disclosure: Parts of this code, especially the inference, have been generated with the assistance of AI*

## Requirements
* Python 3.10 to 3.13
* v4l2loopback
* CUDA 12.*

If you are on an AMD GPU the script may still work through ROCm, currently there are hardcoded values for CUDA. Feel free to send a PR as I don't have an AMD card to test this on.

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

Download an RVM model from the [Github](https://github.com/PeterL1n/RobustVideoMatting#download). Only download models from the torchscript section.

The default location that is check for is `~/rvm_models/rvm_resnet50_fp16.torchscript`, if you want to use a different model or have it in a different location use the `--model` argument to change the path.

### Python
Setup a python virtual environment and install required packages:
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install opencv-python numpy pyvirtualcam
```

## Usage

At its most basic you can run the script without any arguments which will create a blur effect.
```bash
python run.py
```

You can also do a green screen:
```bash
python run.py --bg-mode green
```

Or solid colors:
```bash
python run.py --bg-mode color --bg-color 255,255,255
```

And custom background images:
```bash
python run.py --bg-mode image --bg-image ~/wallpaper.jpg
```

## Mamba
If you had trouble trying to run the script through a virtual environment you can try this approach using Mamba.

Follow this [guide](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to install Mamba. Essentially, install the miniforge package from your distro's package manager.

Then run the following commands to set the environment up:
```bash
mamba create -n pytorch python=3.12
mamba activate pytorch
mamba install -c conda-forge numpy opencv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install pyvirtualcam
```

If you got an error while trying to run the `pip` commands just run `mamba install pip`.

You can now go back to the usage section and re-run the commands.
