use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "cli_tool")] 
struct Args { #[arg(short, long)] name: String }

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("Hello, {}!", args.name);
    Ok(())
}
