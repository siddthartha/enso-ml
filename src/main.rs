mod models;
mod handlers;
mod routes;

use log::{debug, error, log_enabled, info, Level};
use std::net::SocketAddr;

use crate::models::RenderRequest;
use crate::models::HealthcheckResponse;
use crate::handlers::{health_checker_handler, render_handler};
use crate::routes::routes;

#[tokio::main]
async fn main()
{
    pretty_env_logger::init();

    let routes = routes();

    let addr: SocketAddr = "0.0.0.0:80".parse().expect("Incorrect address");

    info!("🔧 Initializing server...");

    let server = warp::serve(routes)
        .bind(addr)
        .await
        .graceful(async move {
            let _ = tokio::signal::ctrl_c()
                .await;
            info!("📛 Shutdown signal received");
        });

    println!("🚀 Enso ML API server started successfully");
    info!("📡 Server listening on http://{}",addr);
    info!("🌐 Local access: http://localhost:{}", addr.port());
    info!("Press Ctrl+C to stop");

    server.run()
        .await;

    info!("✅ Server stopped gracefully");
}
