# enso-ml-net

A PoC of simple asynchronuous json API for running ML-models tasks via Redis queue.

> Pipelines based on `huggingface/candle` ML framework (https://github.com/huggingface/candle)


![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/siddthartha/enso-darknet/rust.yml?logo=rust&label=rust)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/siddthartha/enso-darknet/docker-image.yml)

### Usage with RunPod

* Create and run CPU Pod from official Redis image (`redis:latest` for example)
* Create pod with Enso ML community template (https://runpod.io/console/deploy?template=6b448rr6cb&ref=6isqvo6h)
* Set `ENSO_REDIS_HOST=redis://{REDIS_POD_URL}:{REDIS_POD_EXTERNAL_PORT}` variable in that template
* Now you can put the task to queue:
  * Get `/render/?prompt=some+prompt&steps=25&height=1024&width=768` to start processing
  * Take `uuid` field from response
  * Try to get `/result/{uuid}.jpg` while it becomes ready or try to see intermediatory timesteps like `/result/{uuid}-{step}.jpg`
  * Also any such pod from this template can be tested by hands via simple debug GUI on `https://pod-url/`

### Usage in docker

* `docker pull dogen/enso-ml:latest`
* `docker-compose up -d`
* Run API and server worker with Redis job queue:
  * `wget http://localhost:80/api/render/?prompt=Some%20prompt`

### TODO:

* refactore API to REST-like
* put results to S3
* multiple GPU devices support
* load balancing
* add other various pipelines (Llama, Yolo, etc..)

# You can donate my work on this repository

> USDT/TRC20 address `TWwumLM9UXZbZdW8ySqWNNNkoqvQJ8PMdK`
