use redis::AsyncCommands;
use redis::streams::{StreamReadOptions, StreamReadReply};
use tokio::time::{sleep, Duration};
use redis::{Client, Commands};
use std::string::String;
use candle_core::Tensor;
use log::{debug, error, log_enabled, info, Level};
use redis::Value::{BulkString, SimpleString};
use enso_ml::{
    SD_RENDER_PIPELINE,
    FLUX_RENDER_PIPELINE,
    TASK_PREFIX,
    QUEUE_STREAM_KEY,
    WORKERS_GROUP_NAME,

    models::RenderRequest,
    pipelines::sd::StableDiffusionTask,
    pipelines::sd::StableDiffusionVersion,
    pipelines::flux::FluxTask,
    pipelines::flux::ModelVersion,
    pipelines::Task
};
use enso_ml::pipelines::RenderTask;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if std::env::var("RUST_LOG").is_ok()
    {
        unsafe { std::env::set_var("RUST_LOG", "info"); }
    }

    pretty_env_logger::init();

    let client = Client::open(enso_ml::redis_host()).unwrap();
    let mut connection = client.get_multiplexed_async_connection().await?;

    let _ : Result<(), _> = redis::cmd("XGROUP")
        .arg("CREATE")
        .arg(QUEUE_STREAM_KEY)
        .arg(WORKERS_GROUP_NAME)
        .arg("$")
        .arg("MKSTREAM")
        .query_async(&mut connection)
        .await;

    let worker_name = format!("worker-{}", std::process::id());
    info!("Start worker {worker_name}");

    loop {
        let opts = StreamReadOptions::default()
            .group(WORKERS_GROUP_NAME, &worker_name)
            .block(5000)
            .count(1);

        let reply: StreamReadReply = connection
            .xread_options(&[QUEUE_STREAM_KEY], &[">"], &opts)
            .await?;

        info!("Waiting...");
        if reply.keys.is_empty() {
            sleep(Duration::from_millis(333)).await;
            continue;
        }

        for stream in reply.keys {
            for msg in stream.ids {
                let job_id = msg.id;

                let payload : String = msg.map.get("payload")
                    .and_then(|v| match v {
                        BulkString(bytes) => String::from_utf8(bytes.clone()).ok(),
                        _ => None,
                    }).unwrap_or_default()
                    .to_string();

                info!("📥 Job {job_id}: {payload}");

                let request: RenderRequest = serde_json::from_str(payload.as_str()).unwrap();

                let uuid = request.clone().uuid;
                let pipeline = request.clone().pipeline;
                let seed= request.clone().seed;
                let prompt = request.clone().prompt;
                let steps = request.clone().steps;
                let width = request.clone().width;
                let height = request.clone().height;
                let intermediary_images = request.clone().intermediates;
                let version = request.clone().version;

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
                        intermediary_images,
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
                            1 => ModelVersion::Schnell,
                            2 => ModelVersion::Dev,
                            _ => ModelVersion::Schnell,
                        },
                        use_dmmv: false,
                        seed: Some(seed.clone() as u64),
                    }),
                    _ => {
                        error!("Job with unknown pipeline {pipeline}");
                        continue;
                    }
                };

                // write_connection.set::<String, String, String>(
                //     format!("{}:{uuid}", TASK_PREFIX.to_string()).to_string(),
                //     serde_json::to_string(&task).unwrap()
                // ).unwrap();

                let _: Tensor = task.run(seed.clone())?;


                // confirm job is completed
                let _: () = redis::cmd("XACK")
                    .arg(QUEUE_STREAM_KEY)
                    .arg(WORKERS_GROUP_NAME)
                    .arg(&job_id)
                    .query_async(&mut connection)
                    .await?;

                info!("✅ Job {job_id} completed");
            }
        }
        ()
    }
}
