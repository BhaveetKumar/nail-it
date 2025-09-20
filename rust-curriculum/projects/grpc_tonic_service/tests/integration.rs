use grpc_tonic_service::{echo, start_server, start_server_with_shutdown};
use tokio::task::JoinHandle;

async fn spawn_server() -> (String, JoinHandle<()>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let addr_str = format!("http://{}", addr);
    let handle = tokio::spawn(async move {
        // Ignore server errors on shutdown in test
        let _ = start_server(addr).await;
    });
    // Give the server a moment to bind
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    (addr_str, handle)
}
#[tokio::test]
async fn graceful_shutdown() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        let _ = start_server_with_shutdown(addr, async move { let _ = rx.await; }).await;
    });
    // Let server start
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    // Signal shutdown
    let _ = tx.send(());
    // Ensure task completes
    tokio::time::timeout(std::time::Duration::from_secs(2), handle).await.expect("server did not shut down").unwrap();
}

#[tokio::test]
async fn echo_roundtrip() {
    let (addr, _handle) = spawn_server().await;

    let mut client = echo::echo_client::EchoClient::connect(addr).await.unwrap();
    let req = tonic::Request::new(echo::EchoRequest { message: "hello".into() });
    let resp = client.say(req).await.unwrap().into_inner();
    assert_eq!(resp.message, "hello");
}

#[tokio::test]
async fn health_ok() {
    let (addr, _handle) = spawn_server().await;
    let mut client = echo::echo_client::EchoClient::connect(addr).await.unwrap();
    let req = tonic::Request::new(echo::HealthCheck{});
    let resp = client.health(req).await.unwrap().into_inner();
    assert_eq!(resp.message, "ok");
}
