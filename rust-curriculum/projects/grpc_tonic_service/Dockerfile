FROM rust:1 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /app/target/release/grpc_tonic_service /usr/local/bin/app
EXPOSE 50051
ENTRYPOINT ["/usr/local/bin/app"]
