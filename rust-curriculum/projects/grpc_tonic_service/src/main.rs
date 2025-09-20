use tracing_subscriber::EnvFilter;
use grpc_tonic_service::start_server_with_shutdown;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();
    let addr = "127.0.0.1:50051".parse()?;
    start_server_with_shutdown(addr, async move {
        let _ = tokio::signal::ctrl_c().await;
    }).await?;
    Ok(())
}
