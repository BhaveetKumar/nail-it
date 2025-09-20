FROM rust:1 as builder
RUN cargo install wasm-pack
WORKDIR /app
COPY . .
# Placeholder: project sources would be under ./app
# RUN wasm-pack build --target web

FROM nginx:alpine
COPY ./dist /usr/share/nginx/html
EXPOSE 80
