# Ophto
Repo for all information regarding the ophthalmoscope concept.

### OpenCV (3.1.0) + Python2.7+ Bindings Installation & Setup Guide:

##### 1- Update firmware:-
```
sudo apt-get update
sudo apt-get upgrade
sudo rpi-update
sudo reboot
```
> You can save some disk space by deleting wolfram-engine if you do not use it.

##### 2- Download [this script](https://github.com/istrapid/ophto/blob/master/autoinstall.sh), make it executable, then run it:-
```
$ sudo chmod +x /path/to/autoinstall.sh
$ sudo ./path/to/autoinstall.sh
```

##### 3- To flip the camera output to get a mirror effect and append the following:-
```
$ sudo nano /usr/local/lib/python2.7/dist-packages/imutils/video/pivideostream.py

  class PiVideoStream:
          def __init__(self, resolution=(320, 240), framerate=32, vf=False, hf=False):
                  # initialize the camera and stream
                  self.camera = PiCamera()
                  self.camera.resolution = resolution
                  self.camera.framerate = framerate
                  self.camera.vflip = vf
                  self.camera.hflip = hf
```
> When initializing use PiVideoStream(hf=True).start()

##### NOTE: Installation and setup can take from 75mins to 4hrs. Plan your time accordingly.
