# Based on
# https://github.com/Rust-GPU/Rust-CUDA/blob/8a6cb734d21d5582052fa5b38089d1aa0f4d582f/Dockerfile,
# but updated, slightly fixed to never ask for interactive input during APT
# operations, and with an entrypoint for proper Rust setup as a host user

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -yq \
    build-essential \
    curl xz-utils pkg-config libssl-dev zlib1g-dev libtinfo-dev libxml2-dev

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH $PATH:/root/.cargo/bin

WORKDIR /enso-ml


RUN USER=root cargo new --bin enso-ml

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
    cargo build

# cleanup resources needed for rebuild only
#RUN cargo clean \
#    && rm -rf ${CARGO_HOME}/registry/* \
#    && rm -rf /enso-darknet/libtorch/include

COPY ./data ./data
COPY ./media ./media

# start API server
CMD cargo run
