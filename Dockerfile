FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git cuda-nvcc-12-2 \
    libgl1 libxrender1 curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Install older Intel TBB runtime for the precompiled simulator binary
RUN curl -L -o /tmp/libtbb2.deb \
        http://archive.ubuntu.com/ubuntu/pool/universe/t/tbb/libtbb2_2020.1-2_amd64.deb && \
    dpkg -i /tmp/libtbb2.deb && rm /tmp/libtbb2.deb
RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code to workdir
COPY . .
ENTRYPOINT ["python3", "main_LMI_inference.py"]
