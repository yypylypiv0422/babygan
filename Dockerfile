# ARG AARCH64_BASE_IMAGE=nvidia/cuda:11.4.0-devel-ubuntu18.04
# FROM ${AARCH64_BASE_IMAGE}

# ENV DEBIAN_FRONTEND=noninteractive
# ENV CUDA_CROSS_VERSION=11-4
# ENV CUDA_CROSS_VERSION_DOT=11.4

# FROM pytorch/pytorch:1.15.1-cuda11.4-cudnn8-runtime

# TensorFlow image base written by TensorFlow authors.
# Source: https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/tools/dockerfiles/partials/ubuntu/nvidia.partial.Dockerfile
# -------------------------------------------------------------------------
ARG ARCH=
ARG CUDA=10.1
FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=7.6.4.38-1
ARG CUDNN_MAJOR_VERSION=7
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=6.0.1-1
ARG LIBNVINFER_MAJOR_VERSION=6

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        # There appears to be a regression in libcublas10=10.2.2.89-1 which
        # prevents cublas from initializing in TF. See
        # https://github.com/tensorflow/tensorflow/issues/9489#issuecomment-562394257
        libcublas10=10.2.1.243-1 \ 
        cuda-nvrtc-${CUDA/./-} \
        cuda-cufft-${CUDA/./-} \
        cuda-curand-${CUDA/./-} \
        cuda-cusolver-${CUDA/./-} \
        cuda-cusparse-${CUDA/./-} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig
# -------------------------------------------------------------------------
#
# Custom part
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
    && python -m pip install --upgrade pip \
    && python -m pip install --upgrade \
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
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip3  install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install nvidia-cuda-toolkit -y





EXPOSE 8000
CMD python3 server.py
