use std::string::String;
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

    pub async fn create_group(&mut self) -> Result<(), RedisError>
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

    pub async fn event_loop(&mut self) -> anyhow::Result<()>
    {
        let worker_name = format!("worker-{}", std::process::id());
        info!("start worker {worker_name}");

        let opts = StreamReadOptions::default()
            .group(WORKERS_GROUP_NAME, &worker_name)
            .block(1000) // block 1 sec
            .count(1);


        loop {
            info!("polling job...");
            let reply: StreamReadReply = self.connection
                .xread_options(&[QUEUE_STREAM_KEY], &[">"], &opts)
                .await?;

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
                    let _: Tensor = task.run(seed.clone())?;


                    // confirm job is completed
                    let _: () = redis::cmd("XACK")
                        .arg(QUEUE_STREAM_KEY)
                        .arg(WORKERS_GROUP_NAME)
                        .arg(&job_id)
                        .query_async(&mut self.connection)
                        .await?;

                    info!("✅ Job {job_id} / {uuid} completed");
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

    let mut worker = Worker::new().await;

    let _ : Result<(), _> = worker.create_group().await;

    let _ = worker
        .event_loop();

    Ok(())
}
