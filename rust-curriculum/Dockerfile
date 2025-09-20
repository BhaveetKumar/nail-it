FROM rust:1-bullseye
WORKDIR /work
RUN rustup component add clippy rustfmt
COPY . /work
CMD ["bash", "-lc", "make all"]
