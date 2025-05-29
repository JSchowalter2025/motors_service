FROM alpine

COPY ./requirements.txt /app/requirements/requirements.txt
RUN apk update
RUN apk add --no-cache \
    build-base \
    gcc \
    g++\
    libzmq \
    git \
    musl-dev \
    python3 \
    python3-dev \
    zeromq-dev\
    libftdi1 \
    && cd /app/requirements \
    && pip3 --no-cache-dir install -r requirements.txt \
    && apk del \
    build-base \
    gcc \
    g++\
    musl-dev \
    && rm -rf /var/cache/apk/*

# COPY ./motors_module /app/motors_module
# WORKDIR /app/motors_module
# RUN python3 setup.py install
# RUN cp -r  bellMotors/motor_server /app/motor_server
