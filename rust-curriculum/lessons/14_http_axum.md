---
verified: true
---

# Lesson 14: HTTP with axum

## Overview

Build HTTP services with `axum` over `tokio`, using routers, handlers, and extracts. Test routes with `tower::ServiceExt` without binding sockets.

## Concepts

- Router and routes: `.route("/health", get(handler))`.
- Handlers: async fns returning `impl IntoResponse`.
- Testing: call service with `oneshot` via `ServiceExt::oneshot`.

## Hands-on Exercise

Create a router with `GET /health` → 200 OK and body `"ok"`.

### Cargo.toml

```toml
[package]
name = "lesson14_http_axum"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = { version = "0.7", features = ["macros"] }
tokio = { version = "1.40", features = ["rt", "macros"] }
tower = "0.5"
http = "1"
```

### src/lib.rs

```rust
use axum::{routing::get, Router, response::IntoResponse};
use http::StatusCode;

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

pub fn app() -> Router {
    Router::new().route("/health", get(health))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    #[tokio::test]
    async fn health_ok() {
        let app = app();
        let res = app
            .oneshot(Request::builder().uri("/health").body("").unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
}
```

### Run

```bash
cargo new lesson14_http_axum --lib --vcs none
cd lesson14_http_axum
# replace files
cargo test
```

## Pitfalls

- Use `oneshot` to test handlers without binding ports.
- Initialize tracing to debug route matching if needed.

## Further Reading

- axum: Getting Started — fetched_at: 2025-09-20T00:00:00Z
- tower: Service & layers — fetched_at: 2025-09-20T00:00:00Z
