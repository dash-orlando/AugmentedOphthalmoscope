# Ophto
Repo for all information regarding the ophthalmoscope concept.

### OpenCV + Python2.7+ Bindings Installation & Setup Guide:

##### 1- Update firmware:-
```
$ sudo apt-get update && sudo apt-get upgrade && sudo rpi-update && sudo reboot
```
##### 2- Download essential libraries:-
```
$ sudo apt-get install build-essential cmake pkg-config libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
$ sudo apt-get install libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
$ sudo apt-get install libatlas-base-dev gfortran
$ sudo apt-get install python2.7-dev
```
##### 3- Download pip and numpy:-
```
$ sudo wget https://bootstrap.pypa.io/get-pip.py
$ sudo python get-pip.py
$ sudo pip install numpy
$ sudo pip install --upgrade numpy
```
##### 4- Download OpenCV (2.4.11 version used):-
```
$ sudo wget -O opencv-2.4.11.zip http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.4.11/opencv-2.4.11.zip/download
$ sudo unzip opencv-2.4.11.zip
$ cd opencv-2.4.11/
```
##### 5- Compile and build:-
```
$ sudo mkdir build
$ cd build/
$ sudo cmake -Wno-dev \
	    -D CMAKE_BUILD_TYPE=RELEASE \
	    -D BUILD_TBB=ON \
	    -D WITH_TBB=ON \
	    -D WITH_OPENMP=ON \
	    -D WITH_OPENGL=ON \
	    -D CMAKE_INSTALL_PREFIX=/usr/local \
	    -D BUILD_NEW_PYTHON_SUPPORT=ON \
	    -D INSTALL_C_EXAMPLES=OFF \
	    -D INSTALL_PYTHON_EXAMPLES=ON \
	    -D BUILD_EXAMPLES=ON \
	    -D WITH_FFMPEG=OFF ..

$ sudo make -j4
```

