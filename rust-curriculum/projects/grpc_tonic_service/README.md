# gRPC Service with Tonic

- Prereqs: `protoc` installed (CI uses arduino/setup-protoc).

- Build:

```bash
cargo build -p grpc_tonic_service
```

- Run:

```bash
cargo run -p grpc_tonic_service
```

- Test (automated): `cargo test -p grpc_tonic_service` runs an integration test.

- Test (manual) with grpcurl:

```bash
grpcurl -plaintext -d '{"message":"hello"}' localhost:50051 echo.Echo/Say
```
