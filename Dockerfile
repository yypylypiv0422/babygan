FROM tensorflow/tensorflow:1.15.0-gpu-py3

ARG AARCH64_BASE_IMAGE=nvidia/cuda:11.3.0-devel-ubuntu18.04
FROM ${AARCH64_BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_CROSS_VERSION=11-3
ENV CUDA_CROSS_VERSION_DOT=11.3

FROM base
ARG PYTHON_VERSION=3.7

RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
          python${PYTHON_VERSION} \
          python3-pip \
          python${PYTHON_VERSION}-dev \
# Change default python
    && cd /usr/bin \
    && ln -sf python${PYTHON_VERSION}         python3 \
    && ln -sf python${PYTHON_VERSION}m        python3m \
    && ln -sf python${PYTHON_VERSION}-config  python3-config \
    && ln -sf python${PYTHON_VERSION}m-config python3m-config \
    && ln -sf python3                         /usr/bin/python \
# Update pip and add common packages
    && python -m pip3 install --upgrade pip \
    && python -m pip3 install --upgrade \
        setuptools \
        wheel \
        six \
# Cleanup
    && apt-get clean \
    && rm -rf $HOME/.cache/pip

WORKDIR /

COPY . .


RUN apt-get update && apt-get upgrade -y && apt-get clean

# (...)
RUN apt-get update --fix-missing -y
# Python package management and basic dependencies
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils


# # Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# # Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7
RUN apt update
RUN apt install -y gcc clang clang-tools cmake python3 python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install gdown

Run apt install ninja-build -y

RUN apt-get install build-essential cmake -y
RUN apt-get -y install cmake
RUN pip3 install dlib
RUN mkdir -p raw_images


COPY reqqq.txt .
# RUN python --version
RUN pip3 install -r reqqq.txt
# RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3  install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install nvidia-cuda-toolkit -y





EXPOSE 8000
CMD python3 server.py
