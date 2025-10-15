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
    if std::env::var("RUST_LOG").is_ok()
    {
        unsafe { std::env::set_var("RUST_LOG", "info"); }
    }

    pretty_env_logger::init();

    let routes = routes();

    let listen_host = std::env::var("ENSO_LISTEN_HOST")
        .unwrap_or_else(|_| "0.0.0.0:80".to_string());

    let addr: SocketAddr = listen_host.parse().expect("Incorrect address");

    info!("🔧 Initializing server...");

    let server = warp::serve(routes)
        .bind(addr)
        .await
        .graceful(async move {
            let _ = tokio::signal::ctrl_c()
                .await;
            info!("📛 Shutdown signal received");
        });

    println!("🚀 Enso ML API server started successfully 📡 listening on http://{}", addr);
    println!("⏹ press Ctrl+C to stop");

    server.run()
        .await;

    info!("✅ Server stopped gracefully");
}
