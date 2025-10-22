#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::arch::x86_64::_rdrand64_step;
use uuid::Uuid;

pub mod models;
pub mod pipelines;
pub mod storage;

pub const SD_RENDER_PIPELINE: &str = "StableDiffusion";
pub const FLUX_RENDER_PIPELINE: &str = "Flux";
pub const TASK_PREFIX: &str = "task";
pub const QUEUE_STREAM_KEY: &str = "enso:tasks";
pub const WORKERS_GROUP_NAME: &str = "enso:workers";

pub const GUIDANCE_SCALE: f64 = 7.5;
pub const DEFAULT_STEPS: u8 = 30;
pub const STEPS_LIMIT: u8 = 50;
pub const DEFAULT_REDIS_HOST: &str = "redis://redis:6379/";

pub fn generate_uuid_v4() -> String
{
    // Use hardware random number generation if available, otherwise fall back to OS entropy
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("rdrand") {
            let mut low64_seed: u64 = 0;
            let mut high64_seed: u64 = 0;
            
            unsafe {
                let low_success = _rdrand64_step(&mut low64_seed);
                let high_success = _rdrand64_step(&mut high64_seed);
                
                // Only use hardware random if both calls succeeded
                if low_success == 1 && high_success == 1 {
                    return Uuid::from_u64_pair(low64_seed, high64_seed).to_string();
                }
            }
        }
    }
    
    // Fallback to secure pseudo-random generation from OS entropy
    Uuid::new_v4().to_string()
}

pub fn redis_host() -> String
{
    let redis_host = std::env::var("ENSO_REDIS_HOST")
        .unwrap_or_else(|_| DEFAULT_REDIS_HOST.to_string());

    return redis_host;
}