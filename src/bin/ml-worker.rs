use redis::{Client, Commands};
use std::string::String;
use enso_ml::{
    SD_RENDER_QUEUE,
    FLUX_RENDER_QUEUE,
    TASK_PREFIX,
    models::RenderRequest,
    pipelines::sd::StableDiffusionTask,
    pipelines::sd::StableDiffusionVersion,
    pipelines::flux::FluxTask,
    pipelines::flux::Model,
    pipelines::Task
};
use enso_ml::pipelines::RenderTask;

#[tokio::main]
pub async fn main() -> anyhow::Result<()>
{
    let client = Client::open(enso_ml::redis_host()).unwrap();
    let mut connection = client.get_connection()?;
    let mut write_connection = client.get_connection()?;
    let mut pubsub_connection = connection.as_pubsub();

    pubsub_connection.subscribe(SD_RENDER_QUEUE)?;
    pubsub_connection.subscribe(FLUX_RENDER_QUEUE)?;

    loop {
        let msg = pubsub_connection.get_message()?;
        let payload : String = msg.get_payload()?;
        println!("channel '{}': {}", msg.get_channel_name(), payload);

        let request: RenderRequest = serde_json::from_str(payload.as_str()).unwrap();

        let uuid = request.clone().uuid;
        let seed= request.clone().seed;
        let prompt = request.clone().prompt;
        let steps = request.clone().steps;
        let width = request.clone().width;
        let height = request.clone().height;
        let intermediary_images = request.clone().intermediates;
        let version = request.clone().version;

        write_connection.set::<String, String, String>(
            format!("{}:{uuid}", TASK_PREFIX.to_string()).to_string(),
            payload
        ).unwrap();

        let task = match msg.get_channel_name() {
            SD_RENDER_QUEUE => Task::StableDiffusion(StableDiffusionTask {
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
            FLUX_RENDER_QUEUE => Task::Flux(FluxTask {
                prompt: prompt.clone(),
                cpu: false,
                quantized: false,
                height: Some(height as usize),
                width: Some(width as usize),
                decode_only: None,
                model: match version {
                    1 => Model::Schnell,
                    2 => Model::Dev,
                    _ => Model::Dev,
                },
                use_dmmv: false,
                seed: Some(seed.clone() as u64),
            }),
            _ => panic!("Unknown queue")
        };

        write_connection.set::<String, String, String>(
            format!("{}:{uuid}", TASK_PREFIX.to_string()).to_string(),
            serde_json::to_string(&task).unwrap()
        ).unwrap();

        let _image = task.run(seed.clone())?;

        ()
    }
}