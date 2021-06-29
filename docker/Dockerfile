#
# This Dockerfile installs dependencies for running HERO Radar Odometry.
#
# To use this Dockerfile, use the `docker build` command.
# See https://docs.docker.com/engine/reference/builder/
# for more information.
#

# Ubuntu 20.04 / Python 3.8
FROM nvcr.io/nvidia/cuda:11.1-cudnn8-runtime

# python
RUN apt update
RUN apt upgrade -y --no-install-recommends
RUN apt-get update
RUN apt-get install gcc g++ python3 python3-pip python3-dev libglib2.0-0 libgl1-mesa-glx build-essential libomp-dev -y --no-install-recommends

# Install pip packages
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip
RUN pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install tensorboard
RUN pip3 install -Iv tensorboardx==2.1 matplotlib==3.2.2 opencv-python==4.5.1.48
RUN pip3 install numpy==1.19.4 sklearn Pillow==8.0

# git and wget
RUN apt-get update
RUN apt install git wget -y --no-install-recommends

# cmake
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt install -y libprotobuf-dev protobuf-compiler
RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends tzdata
RUN DEBIAN_FRONTEND=interactive
RUN apt install -q -y --no-install-recommends cmake

# boost
RUN apt install -y libboost-all-dev --no-install-recommends

RUN mkdir workspace

RUN apt install -y libeigen3-dev

# lgmath
RUN cd /workspace && \
 git clone https://github.com/utiasASRL/lgmath.git && \
 cd lgmath && \
# git checkout 979eb07417affc0581d3d69cecae7e3737d04e57 && \
 git checkout feature/eigen3 &&\
 mkdir build && \
 cd build && \
 cmake .. -DUSE_AMENT=OFF && \
 cmake --build . && \
 cmake --install .

# steam
RUN cd /workspace && \
 git clone https://github.com/utiasASRL/steam.git && \
 cd steam && \
# git checkout feature/pointToPointErrorEval && \
 git checkout feature/eigen3 && \
 mkdir build && \
 cd build && \
 cmake .. -DUSE_AMENT=OFF && \
 cmake --build . && \
 cmake --install .

# opencv
RUN apt install -y libopencv-dev python3-opencv --no-install-recommends
