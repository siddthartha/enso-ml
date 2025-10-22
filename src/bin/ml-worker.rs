use std::string::String;
use candle_core::safetensors::save;
use log::{debug, error, log_enabled, info, Level};
use redis::{AsyncCommands, RedisError};
use redis::streams::{StreamReadOptions, StreamReadReply};
use tokio::time::{sleep, Duration};
use redis::{Client};
use redis::Value::{BulkString};
use candle_core::Tensor;
use redis::aio::MultiplexedConnection;

use enso_ml::{
    SD_RENDER_PIPELINE,
    FLUX_RENDER_PIPELINE,
    QUEUE_STREAM_KEY,
    WORKERS_GROUP_NAME,

    models::RenderRequest,
    pipelines::sd::StableDiffusionTask,
    pipelines::sd::StableDiffusionVersion,
    pipelines::flux::FluxTask,
    pipelines::flux::FluxVersion,
    pipelines::Task
};

use enso_ml::pipelines::RenderTask;
use enso_ml::storage;


struct Worker {
    client: Client,
    pub connection: MultiplexedConnection,
}

impl Worker {
    pub async fn new() -> Self
    {
        let client = Client::open(enso_ml::redis_host()).unwrap();
        let connection = client.get_multiplexed_async_connection().await.unwrap();

        Self {
            client,
            connection,
        }
    }

    pub async fn init_redis_group(&mut self) -> Result<(), RedisError>
    {
        redis::cmd("XGROUP")
            .arg("CREATE")
            .arg(QUEUE_STREAM_KEY)
            .arg(WORKERS_GROUP_NAME)
            .arg("$")
            .arg("MKSTREAM")
            .query_async(&mut self.connection)
            .await
    }

    pub async fn pull_job(&mut self) -> StreamReadReply
    {
        let worker_name = self.get_worker_name();
        let opts = StreamReadOptions::default()
            .group(WORKERS_GROUP_NAME, &worker_name)
            .block(1000) // block 1 sec
            .count(1);

        let reply: StreamReadReply = self.connection
            .xread_options(&[QUEUE_STREAM_KEY], &[">"], &opts)
            .await
            .expect("queue read error");

        reply
    }

    pub fn get_worker_name(&self) -> String
    {
        format!("worker-{}", std::process::id())
    }

    pub async fn commit_job(&mut self, job_id: String) -> Result<(), RedisError>
    {
        redis::cmd("XACK")
            .arg(QUEUE_STREAM_KEY)
            .arg(WORKERS_GROUP_NAME)
            .arg(&job_id)
            .query_async(&mut self.connection)
            .await
    }

    pub async fn event_loop(&mut self) -> anyhow::Result<()>
    {
        let worker_name = self.get_worker_name();
        info!("start worker {}", worker_name);

        loop {
            info!("polling job...");
            let reply = self
                .pull_job()
                .await;

            if reply.keys.is_empty() {
                continue;
            }

            for stream in reply.keys.iter() {
                for msg in stream.ids.iter() {
                    let job_id = &msg.id;

                    let payload : String = msg.map.get("payload")
                        .and_then(|v| match v {
                            BulkString(bytes) => String::from_utf8(bytes.clone()).ok(),
                            _ => None,
                        })
                        .unwrap_or_default()
                        .to_string();

                    let request: RenderRequest = serde_json::from_str(payload.as_str()).unwrap();

                    let RenderRequest {
                        uuid,
                        pipeline,
                        seed,
                        prompt,
                        steps,
                        width,
                        height,
                        intermediates,
                        version,
                        ..
                    } = request.clone();


                    let task = match pipeline.as_str() {
                        SD_RENDER_PIPELINE => Task::StableDiffusion(StableDiffusionTask {
                            uuid: uuid.clone().to_string(),
                            prompt: prompt.clone(),
                            height: Some(height as usize),
                            width: Some(width as usize),
                            unet_weights: None,
                            clip_weights: None,
                            vae_weights: None,
                            sliced_attention_size: None,
                            n_steps: steps as usize,
                            seed: seed.clone(),
                            num_samples: 0,
                            final_image: format!("./media/{}.jpg", uuid.clone().to_string()),
                            autocast: false,
                            sd_version: match version {
                                1 => StableDiffusionVersion::V1_5,
                                2 => StableDiffusionVersion::V2_1,
                                3 => StableDiffusionVersion::Xl,
                                4 => StableDiffusionVersion::Turbo,
                                _ => StableDiffusionVersion::V1_5,
                            },
                            intermediary_images: intermediates,
                        }),
                        FLUX_RENDER_PIPELINE => Task::Flux(FluxTask {
                            uuid: uuid.clone().to_string(),
                            prompt: prompt.clone(),
                            cpu: false,
                            quantized: false,
                            height: Some(height as usize),
                            width: Some(width as usize),
                            decode_only: None,
                            model: match version {
                                1 => FluxVersion::Schnell,
                                2 => FluxVersion::Dev,
                                _ => FluxVersion::Schnell,
                            },
                            use_dmmv: false,
                            seed: Some(seed.clone() as u64),
                        }),
                        _ => {
                            error!("Job with unsupported pipeline {pipeline}");
                            continue;
                        }
                    };

                    // write_connection.set::<String, String, String>(
                    //     format!("{}:{uuid}", TASK_PREFIX.to_string()).to_string(),
                    //     serde_json::to_string(&task).unwrap()
                    // ).unwrap();

                    info!("📥 processing job {job_id} / {uuid}:\n{payload}");
                    let result_image: Tensor = match task.run(seed.clone()) {
                        Ok(tensor) => {
                            info!("✅ Job {job_id} / {uuid} completed");
                            tensor
                        },
                        Err(e) => {
                            error!("Job {job_id} / {uuid} failed: {error}", job_id = job_id, uuid = uuid, error = e);
                            continue;
                        }
                    };

                    match storage::save_to_s3_safe(&result_image, uuid.as_str()).await {
                        Ok(tensor) => {
                            info!("✅ Job {job_id} saved", job_id = job_id);
                            tensor
                        },
                        Err(e) => {
                            error!("Job {job_id} / {uuid} failed to save: {error}", job_id = job_id, uuid = uuid, error = e);
                            continue;
                        }
                    };

                    self.commit_job(job_id.clone())
                        .await?;

                }
            }
            ()
        }
    }

}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if std::env::var("RUST_LOG").is_ok()
    {
        unsafe { std::env::set_var("RUST_LOG", "info"); }
    }

    pretty_env_logger::init();

    let mut worker = Worker::new()
        .await;

    let _ = worker
        .init_redis_group()
        .await;

    let _ = worker
        .event_loop()
        .await?;

    Ok(())
}
