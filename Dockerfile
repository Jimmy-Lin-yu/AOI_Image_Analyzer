# 使用 CUDA 12.8 + cuDNN 支援 Blackwell 架構（sm_120）
# FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# 改為官方 PyTorch CUDA 12.8 版（nightly 或 stable 皆可）
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# 安裝其他系統套件
RUN apt-get update && \
    apt-get install -y \
      git wget unzip nano ffmpeg libgl1-mesa-glx libglib2.0-0 \
      python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 下面就不用再反覆 uninstall+install torch 了
COPY requirements.txt /app/requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

COPY . /app
CMD ["bash"]