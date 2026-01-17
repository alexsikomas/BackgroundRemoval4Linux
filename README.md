# RemoveBackground4Linux

*Made because I've given up on waiting for NVIDIA Broadcast to come to Linux.*

## Usage

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

Run using:
```bash
python main.py [model_path] [device_number]
```

Use `--help` to view all the possible options.

*Licensed under GPL-3.0 due to the usage of [linuxpy](https://github.com/tiagocoutinho/linuxpy/)*
