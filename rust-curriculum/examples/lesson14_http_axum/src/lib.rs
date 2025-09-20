use axum::{body::Body, response::IntoResponse, routing::get, Router};
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
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
}
