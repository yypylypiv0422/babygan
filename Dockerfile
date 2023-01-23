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
Run apt install ninja-build -y

# # Upgrade pip to latest version
# RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#     python get-pip.py --force-reinstall && \
#     rm get-pip.py
# Create the environment: \
RUN apt-get install build-essential cmake -y
RUN apt-get -y install cmake
RUN pip3 install dlib

COPY reqqq.txt .
# RUN python --version
RUN pip3 install -r reqqq.txt
# RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3  install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install nvidia-cuda-toolkit -y

# Make RUN commands  use the new environment:
#conda env create -f environment.yaml
#SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]


# Demonstrate the environment is activated:
#RUN conda install -c anaconda flask
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

RUN gdown https://drive.google.com/uc?id=1h7d8X0yWS3IGEeuveNpYsp9ft5fcwkZH
RUN gdown https://drive.google.com/uc?id=1R6sOvuuYP1KNrB6Odq1PkXn8FeHRVE2O
RUN gdown https://drive.google.com/uc?id=1vTbXcTkLlaUUuOy8gf8_3UlLoSxLLZzs
RUN gdown https://drive.google.com/uc?id=1Ze2OLkJ7lmRxx6g5oc701C1DyVn5yUyZ
RUN gdown https://drive.google.com/uc?id=1OVIz5G4WsoJVzjFbyClt4x6WJRTnzwb0
RUN gdown https://drive.google.com/uc?id=1dsoy9JpbJD1J8Wun-_CH8uDvFNEUVrLo

EXPOSE 8000
CMD python3 main.py
