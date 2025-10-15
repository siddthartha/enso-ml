mod models;
mod handlers;
mod routes;

use log::{debug, error, log_enabled, info, Level};

use crate::models::RenderRequest;
use crate::models::HealthcheckResponse;
use crate::handlers::{health_checker_handler, render_handler};
use crate::routes::routes;

#[tokio::main]
async fn main()
{
    println!("🚀 Enso ML API server started successfully");
    warp::serve(routes()).run(([0, 0, 0, 0], 80)).await;
    info!("Gracefully stopped")
}
