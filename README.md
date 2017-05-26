# Ophto
Repo for all information regarding the ophthalmoscope concept.

### OpenCV (3.1.0) + Python2.7+ Bindings Installation & Setup Guide:

##### 1- Update firmware \ remove unnecessary packages:-
```
sudo apt-get update
sudo apt-get purge wolfram* libreoffice*
sudo apt-get upgrade
sudo apt-get dist-upgrade
sudo rpi-update
sudo reboot
```

##### 2- Download required OpenCV packages:-
###### Download essential & optimization libraries:-
```
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg8-dev libjasper-dev libpng12-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libatlas-base-dev gfortran python2.7-dev
sudo apt-get install libgtkglext1 libgtkglext1-dev
```

###### Download/Install pip & numpy:-
```
cd ~
sudo wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip install numpy
sudo pip install --upgrade numpy
```

###### Download OpenCV (ver3.1.0) + extra modules:-
```
cd ~
sudo wget -O opencv.zip https://github.com/opencv/opencv/archive/3.1.0.zip
sudo unzip opencv.zip
sudo wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.1.0.zip
sudo unzip opencv_contrib.zip
```
##### 3- Compile & build:-
> TBB and OpenMP are enabled to improve FPS.
```
cd ~/opencv-3.1.0/
sudo mkdir build
cd build/
sudo cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D BUILD_TBB=ON \
-D WITH_TBB=ON \
-D WITH_OPENMP=ON \
-D WITH_OPENGL=ON \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D BUILD_EXAMPLES=ON \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.1.0/modules ..
```

###### Use 2 cores to insure proper installation and avoid build errors.
```
sudo make -j2
sudo make install
sudo ldconfig
```

##### 3- Enable PiCam/Webcam multithreading (improve FPS) + add mirror effect:-
```
sudo pip install imutils
```

###### Mirror effect:-
```
$ sudo nano /usr/local/lib/python2.7/dist-packages/imutils/video/pivideostream.py

  class PiVideoStream:
          def __init__(self, resolution=(480, 368), framerate=32, vf=False, hf=False):
                  # initialize the camera and stream
                  self.camera = PiCamera()
                  self.camera.resolution = resolution
                  self.camera.framerate = framerate
                  self.camera.vflip = vf
                  self.camera.hflip = hf
```
> When initializing use PiVideoStream(hf=True).start()

##### NOTE: Installation and setup can take from 75mins to 4hrs. Plan your time accordingly.
