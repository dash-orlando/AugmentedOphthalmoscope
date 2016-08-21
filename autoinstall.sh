#!/bin/bash
#OpenCV 3.1.0 + Python2.7+ bindings autoinstall script

#Download essential & optimization libraries:-
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install libatlas-base-dev gfortran python2.7-dev
sudo apt-get install libgtkglext1 libgtkglext1-dev

#Download/Install pip & numpy:-
sudo wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo pip install numpy
sudo pip install --upgrade numpy

#Download OpenCV (ver3.1.0) + extra modules:-
cd ~
sudo wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip
sudo unzip opencv.zip
sudo wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
sudo unzip opencv_contrib.zip

#Compile & build:-
#TBB and OpenMP are enabled to improve FPS.
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
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.1.0/modules \

#Use 2 cores to insure proper installation and avoid build errors.
sudo make -j2
sudo make install
sudo ldconfig


#Download imutils package to enable picam/webcam multithreading:-
#Set desired resolutions by editing pivideostream.py found at
#/usr/local/lib/python2.7/dist-packages/imutils/video/
sudo pip install imutils
