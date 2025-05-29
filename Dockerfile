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

# FROM alpine

# COPY ./requirements.txt /app/requirements/requirements.txt
# RUN apk update
# RUN apk add --no-cache \
#     build-base \
#     gcc \
#     g++\
#     libzmq \
#     git \
#     musl-dev \
#     python3 \
#     python3-dev \
#     zeromq-dev\
#     libftdi1 \
#     && cd /app/requirements \
#     && python3 -m pip --no-cache-dir install -r requirements.txt \
#     && apk del \
#     build-base \
#     gcc \
#     g++\
#     musl-dev \
#     && rm -rf /var/cache/apk/*

# # COPY ./motors_module /app/motors_module
# # WORKDIR /app/motors_module
# # RUN python3 setup.py install
# # RUN cp -r  bellMotors/motor_server /app/motor_server
