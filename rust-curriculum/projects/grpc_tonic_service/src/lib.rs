use tonic::{transport::Server, Request, Response, Status};

pub mod echo {
    tonic::include_proto!("echo");
}

#[derive(Default)]
pub struct EchoService;

#[tonic::async_trait]
impl echo::echo_server::Echo for EchoService {
    async fn say(&self, request: Request<echo::EchoRequest>) -> Result<Response<echo::EchoReply>, Status> {
        let reply = echo::EchoReply { message: request.into_inner().message };
        Ok(Response::new(reply))
    }

    async fn health(&self, _request: Request<echo::HealthCheck>) -> Result<Response<echo::EchoReply>, Status> {
        Ok(Response::new(echo::EchoReply { message: "ok".into() }))
    }
}

pub async fn start_server(addr: std::net::SocketAddr) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let svc = echo::echo_server::EchoServer::new(EchoService::default());
    Server::builder().add_service(svc).serve(addr).await?;
    Ok(())
}
