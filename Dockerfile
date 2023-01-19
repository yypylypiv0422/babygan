#FROM continuumio/miniconda3
ARG AARCH64_BASE_IMAGE=nvidia/cuda:11.4.0-devel-ubuntu18.04
FROM ${AARCH64_BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_CROSS_VERSION=11-4
ENV CUDA_CROSS_VERSION_DOT=11.4
WORKDIR /

COPY . .

RUN apt-get update && apt-get upgrade -y && apt-get clean

# (...)


# Python package management and basic dependencies
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils
RUN apt install --reinstall nvidia-driver-


# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7

# Upgrade pip to latest version

RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

RUN apt-get install build-essential cmake -y
RUN pip install cmake
RUN pip install dlib

#       sudo apt-get install libgtk-3-dev
COPY reqqq.txt .
RUN pip install -r reqqq.txt
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install nvidia-cuda-toolkit -y

# Make RUN commands  use the new environment:
#conda env create -f environment.yaml
#SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]


# Demonstrate the environment is activated:
#RUN conda install -c anaconda flask
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

EXPOSE 8000
CMD python main.py
