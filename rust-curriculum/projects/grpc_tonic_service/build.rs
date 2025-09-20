fn main() {
    let protoc_path = protoc_bin_vendored::protoc_bin_path().expect("protoc not found");
    std::env::set_var("PROTOC", protoc_path);
    let proto = "proto/echo.proto";
    println!("cargo:rerun-if-changed={proto}");
    tonic_build::configure()
        .compile_protos(&[proto], &["proto"]).unwrap();
}
