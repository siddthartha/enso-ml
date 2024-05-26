#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::arch::x86_64::_rdrand64_step;
use serde::{Deserialize, Serialize};
use anyhow::{Error as E, Result};
use uuid::Uuid;

use tokenizers::Tokenizer;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_core::utils::cuda_is_available;
use candle_transformers::models::stable_diffusion;
use stable_diffusion::vae::AutoEncoderKL;
use image::GenericImageView;

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

    /// The height in pixels of the generated image.
    pub height: Option<usize>,

    /// The width in pixels of the generated image.
    pub width: Option<usize>,

    /// The UNet weight file, in .ot or .safetensors format.
    pub unet_weights: Option<String>,

    /// The CLIP weight file, in .ot or .safetensors format.
    pub clip_weights: Option<String>,

    /// The VAE weight file, in .ot or .safetensors format.
    pub vae_weights: Option<String>,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    pub sliced_attention_size: Option<usize>,

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

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq, Deserialize, Serialize)]
pub enum StableDiffusionVersion {
    V1_5,
    V2_1,
    Xl,
    Turbo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

impl StableDiffusionTask {

    pub fn output_filename(
        basename: &str,
        sample_idx: usize,
        num_samples: usize,
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

        let StableDiffusionTask {
            prompt,
            height,
            width,
            n_steps,
            final_image,
            sliced_attention_size,
            sd_version,
            clip_weights,
            vae_weights,
            unet_weights,
            ..
        } = self.clone();

        let cpu = false;
        let num_samples = 1;
        let uncond_prompt = "";
        let tokenizer = None;
        let bsize: usize = 1;
        let use_f16 = false;
        let guidance_scale = None;
        let use_flash_attn = false;
        let img2img: Option<String> = None;
        let img2img_strength = 0.5;

        if !(0. ..=1.).contains(&img2img_strength) {
            anyhow::bail!("img2img-strength should be between 0 and 1, got {img2img_strength}")
        }

        let _guard: Option<bool> = None;

        let guidance_scale = match guidance_scale {
            Some(guidance_scale) => guidance_scale,
            None => match sd_version {
                StableDiffusionVersion::V1_5
                | StableDiffusionVersion::V2_1
                | StableDiffusionVersion::Xl => 7.5,
                StableDiffusionVersion::Turbo => 0.,
            },
        };

        let dtype = if use_f16 { DType::F16 } else { DType::F32 };
        let sd_config = match sd_version {
            StableDiffusionVersion::V1_5 => {
                stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::V2_1 => {
                stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::Xl => {
                stable_diffusion::StableDiffusionConfig::sdxl(sliced_attention_size, height, width)
            }
            StableDiffusionVersion::Turbo => stable_diffusion::StableDiffusionConfig::sdxl_turbo(
                sliced_attention_size,
                height,
                width,
            ),
        };

        let scheduler = sd_config.build_scheduler(n_steps)?;
        let device = candle_examples::device(cpu)?;

        if cuda_is_available()
        {
            device.set_seed(seed as u64)?;
        }

        let use_guide_scale = guidance_scale > 1.0;

        let which = match sd_version {
            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => vec![true, false],
            _ => vec![true],
        };
        let text_embeddings = which
            .iter()
            .map(|first| {
                text_embeddings(
                    &prompt,
                    &uncond_prompt,
                    tokenizer.clone(),
                    clip_weights.clone(),
                    sd_version,
                    &sd_config,
                    use_f16,
                    &device,
                    dtype,
                    use_guide_scale,
                    *first,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
        let text_embeddings = text_embeddings.repeat(vec![bsize, 1, 1])?;
        println!("{text_embeddings:?}");

        println!("Building the autoencoder.");
        let vae_weights = ModelFile::Vae.get(vae_weights, sd_version, use_f16)?;
        let vae = sd_config.build_vae(vae_weights, &device, dtype)?;
        let init_latent_dist = match &img2img {
            None => None,
            Some(image) => {
                let image = image_preprocess(image)?.to_device(&device)?;
                Some(vae.encode(&image)?)
            }
        };
        println!("Building the unet.");
        let unet_weights = ModelFile::Unet.get(unet_weights, sd_version, use_f16)?;
        let unet = sd_config.build_unet(unet_weights, &device, 4, use_flash_attn, dtype)?;

        let t_start = if img2img.is_some() {
            n_steps - (n_steps as f64 * img2img_strength) as usize
        } else {
            0
        };

        let vae_scale = match sd_version {
            StableDiffusionVersion::V1_5
            | StableDiffusionVersion::V2_1
            | StableDiffusionVersion::Xl => 0.18215,
            StableDiffusionVersion::Turbo => 0.13025,
        };

        let mut final_latents = Tensor::randn(0f32, 1., (2, 3), &device)?;

        for idx in 0..num_samples {
            let timesteps = scheduler.timesteps();
            let latents = match &init_latent_dist {
                Some(init_latent_dist) => {
                    let latents = (init_latent_dist.sample()? * vae_scale)?.to_device(&device)?;
                    if t_start < timesteps.len() {
                        let noise = latents.randn_like(0f64, 1f64)?;
                        scheduler.add_noise(&latents, noise, timesteps[t_start])?
                    } else {
                        latents
                    }
                }
                None => {
                    let latents = Tensor::randn(
                        0f32,
                        1f32,
                        vec![bsize, 4, sd_config.height / 8, sd_config.width / 8],
                        &device,
                    )?;
                    // scale the initial noise by the standard deviation required by the scheduler
                    (latents * scheduler.init_noise_sigma())?
                }
            };
            let mut latents = latents.to_dtype(dtype)?;

            println!("starting sampling");
            for (timestep_index, &timestep) in timesteps.iter().enumerate() {
                if timestep_index < t_start {
                    continue;
                }
                let start_time = std::time::Instant::now();
                let latent_model_input = if use_guide_scale {
                    Tensor::cat(&[&latents, &latents], 0)?
                } else {
                    latents.clone()
                };

                println!("scheduler.scale_model_input");
                let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
                println!("unet.forward");
                let noise_pred =
                    unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

                let noise_pred = if use_guide_scale {
                    let noise_pred = noise_pred.chunk(2, 0)?;
                    let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                    (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
                } else {
                    noise_pred
                };

                println!("scheduler.step");
                latents = scheduler.step(&noise_pred, timestep, &latents)?;
                let dt = start_time.elapsed().as_secs_f32();
                println!("step {}/{n_steps} done, {:.2}s", timestep_index + 1, dt);

                if self.intermediary_images {
                    save_image(
                        &vae,
                        &latents,
                        vae_scale,
                        bsize,
                        idx,
                        &final_image,
                        num_samples,
                        Some(timestep_index + 1),
                    )?;
                }
            }

            println!(
                "Generating the final image for sample {}/{}.",
                idx + 1,
                num_samples
            );
            save_image(
                &vae,
                &latents,
                vae_scale,
                bsize,
                idx,
                &final_image,
                num_samples,
                None,
            )?;

            final_latents = latents.clone();
        }

        Ok(final_latents)
    }

}


impl StableDiffusionVersion {
    fn repo(&self) -> &'static str {
        match self {
            Self::Xl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::Turbo => "stabilityai/sdxl-turbo",
        }
    }

    fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
        }
    }

    fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
        }
    }
}

impl ModelFile {
    fn get(
        &self,
        filename: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
    ) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        let tokenizer_repo = match version {
                            StableDiffusionVersion::V1_5 | StableDiffusionVersion::V2_1 => {
                                "openai/clip-vit-base-patch32"
                            }
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => {
                                // This seems similar to the patch32 version except some very small
                                // difference in the split regex.
                                "openai/clip-vit-large-patch14"
                            }
                        };
                        (tokenizer_repo, "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => (version.repo(), version.clip_file(use_f16)),
                    Self::Clip2 => (version.repo(), version.clip2_file(use_f16)),
                    Self::Unet => (version.repo(), version.unet_file(use_f16)),
                    Self::Vae => {
                        // Override for SDXL when using f16 weights.
                        // See https://github.com/huggingface/candle/issues/1060
                        if matches!(
                            version,
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo,
                        ) && use_f16
                        {
                            (
                                "madebyollin/sdxl-vae-fp16-fix",
                                "diffusion_pytorch_model.safetensors",
                            )
                        } else {
                            (version.repo(), version.vae_file(use_f16))
                        }
                    }
                };

                // let cache = Cache::new("./data".into());
                // let cache = cache.repo(hf_hub::Repo::new(repo.to_string(), hf_hub::RepoType::Model));
                // let filename = cache.get(path).unwrap();

                let filename = Api::new()?.model(repo.to_string()).get(path)?;

                Ok(filename)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn save_image(
    vae: &AutoEncoderKL,
    latents: &Tensor,
    vae_scale: f64,
    bsize: usize,
    idx: usize,
    final_image: &str,
    num_samples: usize,
    timestep_ids: Option<usize>,
) -> Result<()>
{
    let images = vae.decode(&(latents / vae_scale)?)?;
    let images = ((images / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    for batch in 0..bsize {
        let image = images.i(batch)?;
        let image_filename = StableDiffusionTask::output_filename(
            final_image,
            (bsize * idx) + batch + 1,
            batch + num_samples,
            timestep_ids,
        );
        candle_examples::save_image(&image, image_filename)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings(
    prompt: &str,
    uncond_prompt: &str,
    tokenizer: Option<String>,
    clip_weights: Option<String>,
    sd_version: StableDiffusionVersion,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    use_f16: bool,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
    first: bool,
) -> Result<Tensor>
{
    let tokenizer_file = if first {
        ModelFile::Tokenizer
    } else {
        ModelFile::Tokenizer2
    };
    let tokenizer = tokenizer_file.get(tokenizer, sd_version, use_f16)?;
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    println!("Running with prompt \"{prompt}\".");
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    if tokens.len() > sd_config.clip.max_position_embeddings {
        anyhow::bail!(
            "the prompt is too long, {} > max-tokens ({})",
            tokens.len(),
            sd_config.clip.max_position_embeddings
        )
    }
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    println!("Building the Clip transformer.");
    let clip_weights_file = if first {
        ModelFile::Clip
    } else {
        ModelFile::Clip2
    };
    let clip_weights = clip_weights_file.get(clip_weights, sd_version, false)?;
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().unwrap()
    };
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer
            .encode(uncond_prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        if uncond_tokens.len() > sd_config.clip.max_position_embeddings {
            anyhow::bail!(
                "the negative prompt is too long, {} > max-tokens ({})",
                uncond_tokens.len(),
                sd_config.clip.max_position_embeddings
            )
        }
        while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
            uncond_tokens.push(pad_id)
        }

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}

fn image_preprocess<T: AsRef<std::path::Path>>(path: T) -> anyhow::Result<Tensor>
{
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    let height = height - height % 32;
    let width = width - width % 32;
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
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