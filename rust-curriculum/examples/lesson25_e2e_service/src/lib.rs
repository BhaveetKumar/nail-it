use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::{get, post}, Json, Router};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::{Arc, RwLock}};
use tracing::instrument;

#[derive(Clone, Default)]
pub struct AppState { store: Arc<RwLock<HashMap<String, Item>>> }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", content = "data")]
pub enum Item {
    Text(String),
    Binary(#[serde(with = "b64")] Vec<u8>),
}

pub mod b64 {
    use base64::{engine::general_purpose, Engine};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &Vec<u8>, s: S) -> Result<S::Ok, S::Error> {
        let encoded = general_purpose::STANDARD.encode(bytes);
        String::serialize(&encoded, s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let s = String::deserialize(d)?;
        general_purpose::STANDARD
            .decode(s.as_bytes())
            .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("not found")] NotFound,
    #[error("serde error: {0}")] Serde(#[from] serde_json::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        match self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, "not found").into_response(),
            ApiError::Serde(e) => (StatusCode::BAD_REQUEST, format!("serde error: {e}")).into_response(),
        }
    }
}

#[instrument(skip(state))]
pub async fn put_item(State(state): State<AppState>, axum::extract::Path(key): axum::extract::Path<String>, Json(item): Json<Item>) -> Result<StatusCode, ApiError> {
    state.store.write().unwrap().insert(key, item);
    Ok(StatusCode::CREATED)
}

#[instrument(skip(state))]
pub async fn get_item(State(state): State<AppState>, axum::extract::Path(key): axum::extract::Path<String>) -> Result<Json<Item>, ApiError> {
    let store = state.store.read().unwrap();
    let item = store.get(&key).cloned().ok_or(ApiError::NotFound)?;
    Ok(Json(item))
}

pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(|| async { StatusCode::OK }))
        .route("/items/:key", post(put_item).get(get_item))
        .with_state(state)
}

pub fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body, body::Body, http::{Request, StatusCode}};
    use serde_json::json;
    use tower::util::ServiceExt;
    use base64::Engine;

    #[tokio::test]
    async fn health_ok() {
        init_tracing();
        let app = app(AppState::default());
        let resp = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn put_and_get_text_item() {
        init_tracing();
        let app = app(AppState::default());
        let body = json!({"type":"Text","data":"hello"}).to_string();
        let resp = app.clone()
            .oneshot(Request::builder().method("POST").uri("/items/greeting").header("content-type","application/json").body(Body::from(body)).unwrap())
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let resp = app
            .oneshot(Request::builder().uri("/items/greeting").body(Body::empty()).unwrap())
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

    let bytes = body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let item: Item = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(item, Item::Text("hello".into()));
    }

    #[tokio::test]
    async fn put_and_get_binary_item() {
        init_tracing();
        let app = app(AppState::default());
    let data = base64::engine::general_purpose::STANDARD.encode(&[1u8,2,3,4,255]);
        let body = json!({"type":"Binary","data":data}).to_string();
        let resp = app.clone()
            .oneshot(Request::builder().method("POST").uri("/items/bin").header("content-type","application/json").body(Body::from(body)).unwrap())
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        let resp = app
            .oneshot(Request::builder().uri("/items/bin").body(Body::empty()).unwrap())
            .await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

    let bytes = body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let item: Item = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(item, Item::Binary(vec![1,2,3,4,255]));
    }
}
