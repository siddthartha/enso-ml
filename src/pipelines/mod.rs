use std::path::PathBuf;
use anyhow::{Error, Result};
use candle_core::Tensor;
use crate::pipelines::flux::FluxTask;
use crate::pipelines::sd::StableDiffusionTask;
use serde::{Serialize, Deserialize};

pub trait RenderTask {
    fn run(&self, seed: i64) -> Result<Tensor, Error>;
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "pipeline_type")]
pub enum Task {
    StableDiffusion(sd::StableDiffusionTask),
    Flux(flux::FluxTask),
}

pub mod sd;
pub mod flux;

impl RenderTask for Task {
    fn run(&self, seed: i64) -> Result<Tensor, Error> {
        match self {
            Task::StableDiffusion(task) => task.run(seed),
            Task::Flux(task) => task.run(seed),
        }
    }
}

impl Task {
    pub fn get_output_filename(uuid: String) -> String {
        format!("./media/{}.jpg", uuid.clone())
    }

    pub fn as_sd(&self) -> Option<&StableDiffusionTask> {
        match self {
            Task::StableDiffusion(task) => Some(task),
            _ => None,
        }
    }

    pub fn as_flux(&self) -> Option<&FluxTask> {
        match self {
            Task::Flux(task) => Some(task),
            _ => None,
        }
    }

    pub fn as_sd_mut(&mut self) -> Option<&mut StableDiffusionTask> {
        match self {
            Task::StableDiffusion(task) => Some(task),
            _ => None,
        }
    }

    pub fn as_flux_mut(&mut self) -> Option<&mut FluxTask> {
        match self {
            Task::Flux(task) => Some(task),
            _ => None,
        }
    }

    pub fn unwrap_sd(&self) -> &StableDiffusionTask {
        match self {
            Task::StableDiffusion(task) => task,
            _ => panic!("Not a StableDiffusion task"),
        }
    }

    pub fn unwrap_flux(&self) -> &FluxTask {
        match self {
            Task::Flux(task) => task,
            _ => panic!("Not a Flux task"),
        }
    }
}

pub fn get_storage_path() -> PathBuf {
    PathBuf::from(std::env::var("ENSO_HF_HOME")
        .unwrap_or("./data".into()))
}