use grpc_tonic_service::echo;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut client = echo::echo_client::EchoClient::connect("http://127.0.0.1:50051").await?;
    let resp = client.say(tonic::Request::new(echo::EchoRequest{ message: "hello".into() })).await?;
    println!("{:?}", resp.into_inner().message);
    Ok(())
}