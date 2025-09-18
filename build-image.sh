#!/bin/bash

docker build -t dogen/enso-ml:latest . --build-arg CUDA_COMPUTE_CAP="75,80,86"