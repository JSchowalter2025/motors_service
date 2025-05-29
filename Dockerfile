# Dockerfile
FROM python:3.9-slim
ENV DEBIAN_FRONTEND=noninteractive 
# install ZMQ and bash
RUN apt-get update && \
    apt-get install -y \
    bash \
    python3-pip \
    libtool pkg-config build-essential autoconf automake \
    libzmq3-dev \
    libftdi1 \
    git \ 
    # apt-get install -y --no-install-recommends bash && \
    # pip install pyzmq && \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN python3 -m pip install --upgrade pip && \
     pip3 --no-cache-dir install -r requirements.txt
