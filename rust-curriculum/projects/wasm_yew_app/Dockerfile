FROM rust:1 as builder
RUN curl -sSfL https://rustwasm.github.io/wasm-pack/installer/init.sh | sh
WORKDIR /app
COPY Cargo.toml .
COPY src ./src
COPY index.html ./
RUN wasm-pack build --target web --release

FROM nginx:1.27-alpine
WORKDIR /usr/share/nginx/html
COPY --from=builder /app/index.html ./
COPY --from=builder /app/pkg ./pkg
COPY nginx.conf /etc/nginx/conf.d/default.conf
RUN adduser -D -H -u 10001 appuser \
	&& chown -R appuser:appuser /usr/share/nginx/html
USER 10001
EXPOSE 8080
