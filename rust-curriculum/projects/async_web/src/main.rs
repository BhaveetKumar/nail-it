use hyper::{service::{make_service_fn, service_fn}, Body, Request, Response, Server};
use tracing_subscriber::EnvFilter;

async fn handle(_req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    Ok(Response::new(Body::from("hello")))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();
    let addr = ([127,0,0,1], 8080).into();
    let make_svc = make_service_fn(|_conn| async { Ok::<_, hyper::Error>(service_fn(handle)) });
    let server = Server::bind(&addr).serve(make_svc);
    println!("listening on http://{}", addr);
    server.await?;
    Ok(())
}
