use log::{debug, error, log_enabled, info, Level};
use std::collections::HashMap;
use warp::Filter;
use crate::{health_checker_handler, render_handler};

pub fn routes() -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    let log = warp::log::custom(|info| {
        info!(
            "{} {} {} {} in {:?}",
            info.remote_addr().map(|addr| addr.to_string()).unwrap_or_else(|| "unknown".to_string()),
            info.method(),
            info.path(),
            info.status().as_u16(),
            info.elapsed()
        );
    });

    let health_checker_route = warp::path!("api" / "health")
        .and(warp::get())
        .and_then(health_checker_handler);

    let render_route = warp::path!("api" / "render")
        .and(warp::query::<HashMap<String, String>>())
        .and(warp::get())
        .and_then(render_handler);

    let root_route = warp::path::end()
        .and(warp::get())
        .and(warp::fs::file("./gui/index.html"));

    let download_route = warp::path("result")
        .and(warp::fs::dir("./media/"));

    let gui_route = warp::path("gui")
        .and(warp::fs::dir("./gui/"));

    let api = health_checker_route
        .or(render_route)
        .or(download_route)
        .or(gui_route)
        .or(root_route)
        .with(log);

    api
}