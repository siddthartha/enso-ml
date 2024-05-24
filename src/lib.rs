use std::arch::x86_64::_rdrand64_step;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub mod models;

pub const GUIDANCE_SCALE: f64 = 7.5;
pub const DEFAULT_STEPS: u8 = 30;
pub const STEPS_LIMIT: u8 = 50;
pub const RENDER_QUEUE: &str = "render";
pub const TASK_PREFIX: &str = "task";

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct StableDiffusionTask {
    /// The prompt to be used for image generation.
    pub prompt: String,

    /// When set, use the CPU for the listed devices, can be 'all', 'unet', 'clip', etc.
    /// Multiple values can be set.
    pub cpu: Vec<String>,

    /// The height in pixels of the generated image.
    pub height: Option<i64>,

    /// The width in pixels of the generated image.
    pub width: Option<i64>,

    /// The UNet weight file, in .ot or .safetensors format.
    pub unet_weights: Option<String>,

    /// The CLIP weight file, in .ot or .safetensors format.
    pub clip_weights: Option<String>,

    /// The VAE weight file, in .ot or .safetensors format.
    pub vae_weights: Option<String>,

    /// The file specifying the vocabulary to used for tokenization.
    pub vocab_file: String,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    pub sliced_attention_size: Option<i64>,

    /// The number of steps to run the diffusion for.
    pub n_steps: usize,

    /// The random seed to be used for the generation. Default is 0, means generate random value.
    pub seed: i64,

    /// The number of samples to generate.
    pub num_samples: i64,

    /// The name of the final image to generate.
    pub final_image: String,

    /// Use autocast (disabled by default as it may use more memory in some cases).
    pub autocast: bool,

    pub sd_version: StableDiffusionVersion,

    /// Generate intermediary images at each step.
    pub intermediary_images: bool,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, clap::ValueEnum)]
pub enum StableDiffusionVersion {
    V1_5,
    V2_1,
}

impl StableDiffusionTask {

    pub fn output_filename(
        basename: &str,
        sample_idx: i64,
        num_samples: i64,
        timestep_idx: Option<usize>,
    ) -> String {
        let filename = if num_samples > 1 {
            match basename.rsplit_once('.') {
                None => format!("{basename}.{sample_idx}.jpg"),
                Some((filename_no_extension, extension)) => {
                    format!("{filename_no_extension}.{sample_idx}.{extension}")
                }
            }
        } else {
            basename.to_string()
        };
        match timestep_idx {
            None => filename,
            Some(timestep_idx) => match filename.rsplit_once('.') {
                None => format!("{filename}-{timestep_idx}.jpg"),
                Some((filename_no_extension, extension)) => {
                    format!("{filename_no_extension}-{timestep_idx}.{extension}")
                }
            },
        }
    }

    pub fn run(&self, seed: i64) -> Result<Tensor, anyhow::Error>
    {

//         let StableDiffusionTask {
//             prompt,
//             cpu,
//             height,
//             width,
//             n_steps,
// //        seed,
//             vocab_file,
//             final_image,
//             sliced_attention_size,
//             num_samples,
//             sd_version,
//             ..
//         } = self.clone();
//
//
//         return Ok(image);
        let device = Device::Cpu;

        let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
        let b = Tensor::randn(0f32, 1., (3, 4), &device)?;

        let c = a.matmul(&b)?;
        println!("{c}");
        println!("{seed}");

        return Ok(c);
    }

}

pub fn generate_uuid_v4() -> String
{
    let mut low64_seed: u64 = 0;
    let mut high64_seed: u64 = 0;

    unsafe {
        _rdrand64_step(&mut low64_seed);
        _rdrand64_step(&mut high64_seed);
    }

    let uuid = Uuid::from_u64_pair(low64_seed, high64_seed).to_string().clone();
    uuid
}

pub fn redis_host() -> String
{
    let _redis_host = std::env::var_os("ENSO_REDIS_HOST");

    let redis_host = match _redis_host {
        None => "redis://redis:6379/".to_string(),
        Some(redis_host_var) => {
            let redis_host = redis_host_var.into_string().unwrap();
            redis_host.clone()
        },
    };

    return redis_host;
}