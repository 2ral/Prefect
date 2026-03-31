FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu22.04

LABEL maintainer="turalevro@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

RUN mkdir -p /root/.cache \
    && mkdir -p /root/datasets \
    && mkdir -p /root/src \
    && mkdir -p /root/scripts

COPY LICENSE.txt /root/
COPY pyproject.toml /root/
COPY src/ /root/src/
COPY scripts/ /root/scripts/

RUN apt update && apt upgrade -y && apt install -y \
    python3 \
    python3-pip \
    nano \
    zip \
    unzip \
    && apt clean \
    && apt autoremove --purge -y \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --upgrade setuptools \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
    && pip install --no-cache-dir bottle==0.13.2 urllib3==2.0.5 msticpy gcovr mo-sql-parsing \
    && (rm -rf /root/.cache/* || true) \
    && (rm -rf /root/src/modelizer.egg-info || true)

RUN chmod +x /root/src/modelizer/generators/implementations/coverage/bc/setup.sh \
    && /root/src/modelizer/generators/implementations/coverage/bc/setup.sh
RUN chmod +x /root/src/modelizer/generators/implementations/coverage/grep/setup.sh \
    && /root/src/modelizer/generators/implementations/coverage/grep/setup.sh
RUN chmod +x /root/src/modelizer/generators/implementations/coverage/re2/full_setup_standalone.sh \
    && /root/src/modelizer/generators/implementations/coverage/re2/full_setup_standalone.sh
