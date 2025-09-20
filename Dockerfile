FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -yq \
    build-essential \
    curl xz-utils pkg-config libssl-dev zlib1g-dev libtinfo-dev libxml2-dev

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV CUDA_ROOT=/usr/local/cuda
ENV PATH=/root/.cargo/bin:$PATH
ENV PATH=$CUDA_ROOT/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN USER=root cargo new --bin enso-ml
WORKDIR /enso-ml

RUN apt-get update \
    && apt-get install -y cmake libclang-dev gcc libc-bin libc-dev-bin libc6 \
    && apt-get install -y tini \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf ./src

COPY ./Cargo.toml ./Cargo.toml
COPY ./src ./src

ENV PATH=/enso-ml:${PATH}

RUN \
    ln -s -r ./target/release/enso-ml ./enso-ml \
    && ln -s -r ./target/release/ml-worker ./ml-worker

# Принудительно указать compute capabilities
#ENV CUDA_COMPUTE_CAP="75,80,86"
#ENV CUDA_ARCH_LIST="7.5;8.0;8.6"

# Установить фейковый nvidia-smi для build-time
#RUN echo '#!/bin/bash\necho "GPU 0: Tesla V100 (UUID: GPU-fake)"' > /usr/bin/nvidia-smi && \
#    chmod +x /usr/bin/nvidia-smi

#RUN cargo build --release

# cleanup resources needed for rebuild only
#RUN cargo clean \
#    && rm -rf ${CARGO_HOME}/registry/* \
#    && rm -rf /enso-ml/libtorch/include

COPY ./data ./data
COPY ./media ./media
COPY ./gui ./gui

# start ML worker and API server
CMD ["sh", "-c", "(cargo build --release) && (./ml-worker & ./enso-ml)"]

