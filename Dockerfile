FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -yq \
    build-essential \
    curl xz-utils pkg-config libssl-dev zlib1g-dev libtinfo-dev libxml2-dev

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH $PATH:/root/.cargo/bin

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

# CAN NOT be built on system without real GPU installed!
# so we build before run on cloud side! :/

RUN \
    cargo build --release \
    && cp /enso-ml/target/release/enso-ml ./enso-ml \
    && cp /enso-ml/target/release/ml-worker ./ml-worker

# cleanup resources needed for rebuild only
#RUN cargo clean \
#    && rm -rf ${CARGO_HOME}/registry/* \
#    && rm -rf /enso-ml/libtorch/include

COPY ./data ./data
COPY ./media ./media

# start API server
CMD (./ml-worker & ./enso-darknet)

