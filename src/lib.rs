#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use std::arch::x86_64::_rdrand64_step;
use uuid::Uuid;

pub mod models;
pub mod pipelines;

pub const SD_RENDER_QUEUE: &str = "render:sd";
pub const FLUX_RENDER_QUEUE: &str = "render:flux";
pub const TASK_PREFIX: &str = "task";

pub const GUIDANCE_SCALE: f64 = 7.5;
pub const DEFAULT_STEPS: u8 = 30;
pub const STEPS_LIMIT: u8 = 50;

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
    let redis_host = std::env::var("ENSO_REDIS_HOST")
        .unwrap_or_else(|_| "redis://redis:6379/".to_string());

    return redis_host;
}